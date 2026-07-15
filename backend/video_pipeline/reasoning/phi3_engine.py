"""
video_pipeline/reasoning/phi3_engine.py
==========================================
Phi-3-mini-4k-instruct reasoning engine — GPU 4-bit quantization ONLY.

RTX 3050 (4 GB VRAM) setup
----------------------------
  device="cuda"  +  load_in_4bit=True  →  ~2.5 GB VRAM  ✓ fits
  No CPU fallback. No FP16 offload. No subprocess probe.
  If CUDA is unavailable or loading fails, a RuntimeError is raised immediately.

Fix history
-----------
v2.0.1  — correct tokenizer, EOS tokens, chat template, add_special_tokens
v2.0.2  — trust_remote_code=False on model (native Phi3ForCausalLM)
v2.0.3  — attn_implementation=eager (no flash-attn / rope_scaling KeyError)
v2.0.4  — dtype instead of torch_dtype (deprecation fix)
v2.0.5  — max_length fix: cap max_new_tokens to ctx_len//2 before tokenising
v2.1.0  — GPU 4-bit ONLY: removed subprocess probe, FP16 offload, CPU fallback
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        PreTrainedTokenizerFast,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "[Phi3Reasoner] transformers/torch not installed.\n"
        "Install: pip install transformers torch accelerate bitsandbytes safetensors tokenizers"
    )

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Context window for Phi-3-mini-4k-instruct.
import os as _os
_PHI3_CTX_LEN = int(_os.environ.get("PHI3_CONTEXT_LENGTH", "4096"))

try:
    from transformers import StoppingCriteriaList
    STOPPING_CRITERIA_AVAILABLE = True
except ImportError:
    STOPPING_CRITERIA_AVAILABLE = False
    StoppingCriteriaList = None   # type: ignore[assignment,misc]


class _JsonCompleteStopper:
    """
    Callable stopping criterion — duck-types into StoppingCriteriaList
    without subclassing StoppingCriteria.

    Space-decode fix: decodes the full new-token suffix on every step
    (input_ids[:, prompt_len:]) rather than a single last token, so
    inter-token whitespace is preserved and words are not concatenated.
    """

    def __init__(self, tokenizer: Any, owner: Any = None) -> None:
        self._tok         = tokenizer
        self._owner       = owner
        self._done        = False
        self._prompt_len: int = 0

    def __call__(self, input_ids: Any, scores: Any, **kwargs) -> bool:
        if self._done:
            return True

        try:
            new_ids = input_ids[0, self._prompt_len:]
            if new_ids.shape[0] == 0:
                return False
            text = self._tok.decode(new_ids, skip_special_tokens=True)
        except Exception:
            return False

        if self._owner is not None:
            self._owner._last_raw_output = text

        depth   = 0
        started = False
        in_str  = False
        escape  = False
        for c in text:
            if escape:
                escape = False
                continue
            if c == '\\' and in_str:
                escape = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c in ('{', '['):
                depth += 1
                started = True
            elif c in ('}', ']'):
                depth -= 1
                if started and depth <= 0:
                    self._done = True
                    return True
        return False


def _build_phi3_prompt(user_message: str, system_message: str) -> str:
    """Manual Phi-3 chat template — fallback if apply_chat_template unavailable."""
    return (
        f"<|system|>\n{system_message}<|end|>\n"
        f"<|user|>\n{user_message}<|end|>\n"
        f"<|assistant|>\n"
    )


class Phi3Reasoner:
    """
    Wraps Phi-3-mini-4k-instruct for academic content extraction.

    Loads EXCLUSIVELY on GPU with 4-bit NF4 bitsandbytes quantization.
    Raises RuntimeError at construction time if CUDA is not available
    or if loading fails — there is no silent CPU/FP16 fallback.

    Drop-in replacement for LlamaReasoner:
      reason(prompt)      -> dict | list
      reason_text(prompt) -> str
    """

    _SYSTEM_JSON = (
        "You are an academic content extraction assistant. "
        "You always respond with valid JSON only. "
        "Never include markdown fences, preamble, or explanation — "
        "output the JSON object or array and nothing else."
    )
    _SYSTEM_TEXT = (
        "You are an expert academic note-taker and educator. "
        "You produce clear, well-structured Markdown that students can use to study."
    )

    def __init__(
        self,
        model_id:       str           = "models/phi3mini",
        max_new_tokens: int           = 1024,
        device:         str           = "cpu",   # accepted but ignored — always uses cuda
        load_in_4bit:   bool          = True,    # accepted but ignored — always True
        temperature:    float         = 0.1,
        top_p:          float         = 0.95,
        do_sample:      bool          = False,
        adapter_path:   Optional[str] = None,
    ) -> None:
        self.model_id       = model_id
        self.max_new_tokens = max_new_tokens
        self.device         = "cuda"   # always GPU — hard-coded
        self.temperature    = temperature
        self.top_p          = top_p
        self.do_sample      = do_sample
        self.adapter_path   = adapter_path

        self._model:           Any = None
        self._tokenizer:       Any = None
        self._stop_token_ids:  Optional[List[int]] = None
        self._last_raw_output: str = ""

        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "[Phi3Reasoner] transformers/torch not installed. "
                "Run: pip install transformers torch accelerate bitsandbytes"
            )

        self._load_model()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def context_length(self) -> int:
        return _PHI3_CTX_LEN

    def _safe_new_tokens(self, prompt_token_len: int) -> int:
        """
        Return the largest safe max_new_tokens given the prompt length.
        Always leaves at least 64 tokens of headroom so max_length > 0.
        """
        available = _PHI3_CTX_LEN - prompt_token_len - 64
        safe = max(64, min(self.max_new_tokens, available))
        if safe < self.max_new_tokens:
            logger.debug(
                f"[Phi3Reasoner] max_new_tokens capped: "
                f"requested={self.max_new_tokens} prompt_tokens={prompt_token_len} "
                f"ctx={_PHI3_CTX_LEN} -> safe={safe}"
            )
        return safe

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Load Phi-3-mini with 4-bit NF4 bitsandbytes quantization on CUDA.

        VRAM budget (RTX 3050, 4 GB):
          Phi-3-mini 4-bit NF4   ~2.5 GB
          EasyOCR (CPU mode)       0 GB   (pipeline forces OCR to CPU)
          faster-whisper tiny     ~0.3 GB (released before this runs)
          OS + driver             ~0.3 GB
          ─────────────────────────────
          Total                  ~3.1 GB  ✓ fits with ~0.9 GB headroom

        Raises RuntimeError if:
          - CUDA is not available
          - bitsandbytes is not installed
          - model loading fails for any reason
        """
        # ── Guard: CUDA must be available ─────────────────────────────────────
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[Phi3Reasoner] CUDA is not available. "
                "Phi-3 requires a CUDA GPU with 4-bit bitsandbytes support. "
                "Set CUDA_VISIBLE_DEVICES or install the correct CUDA toolkit."
            )

        # ── Guard: bitsandbytes must be importable ─────────────────────────────
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "[Phi3Reasoner] bitsandbytes is not installed. "
                "Run: pip install bitsandbytes"
            )

        device_name = torch.cuda.get_device_name(0)
        vfree, vtotal = torch.cuda.mem_get_info(0)
        logger.info(
            f"[Phi3Reasoner] Loading 4-bit model on CUDA: {device_name} | "
            f"VRAM {vfree/(1024**3):.1f}GB free of {vtotal/(1024**3):.1f}GB total"
        )

        # ── Cleanup before load ───────────────────────────────────────────────
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # ── Tokenizer ─────────────────────────────────────────────────────────
        logger.info(f"[Phi3Reasoner] Loading tokenizer from: {self.model_id}")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
        except Exception as exc:
            logger.warning(
                f"[Phi3Reasoner] AutoTokenizer failed ({exc}), "
                "falling back to PreTrainedTokenizerFast"
            )
            self._tokenizer = PreTrainedTokenizerFast.from_pretrained(
                self.model_id, trust_remote_code=True,
            )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        logger.info(f"[Phi3Reasoner] Tokenizer ready: {type(self._tokenizer).__name__}")

        # ── Stop token IDs ────────────────────────────────────────────────────
        stop_ids: List[int] = []
        eos_id = self._tokenizer.eos_token_id
        if eos_id is not None:
            stop_ids.append(eos_id)
            logger.info(f"[Phi3Reasoner] EOS '{self._tokenizer.eos_token}' id={eos_id}")
        end_id = self._tokenizer.convert_tokens_to_ids("<|end|>")
        if (
            end_id is not None
            and end_id != self._tokenizer.unk_token_id
            and end_id not in stop_ids
        ):
            stop_ids.append(end_id)
            logger.info(f"[Phi3Reasoner] Stop '<|end|>' id={end_id}")
        self._stop_token_ids = stop_ids if stop_ids else None

        # ── 4-bit quantization config ─────────────────────────────────────────
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,   # nested quantization saves ~0.1 GB extra
        )
        logger.info(
            "[Phi3Reasoner] Quantization: 4-bit NF4 + double quant, "
            "compute dtype=float16"
        )

        # ── Model load ────────────────────────────────────────────────────────
        logger.info(f"[Phi3Reasoner] Loading model from: {self.model_id}")
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                attn_implementation="eager",    # no flash-attn dependency
                quantization_config=quant_config,
                device_map="auto",
                max_memory={
                    0 :   "3.0GiB",   # GPU — leaves headroom for KV cache + OCR
                    "cpu": "1GiB",     # minimal CPU shard space only (no offload)
                },
                dtype=torch.float16,
            )
        except Exception as exc:
            gc.collect()
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"[Phi3Reasoner] 4-bit model load failed: {exc}\n"
                "Ensure bitsandbytes is installed and CUDA toolkit matches PyTorch: "
                "pip install bitsandbytes --upgrade"
            ) from exc

        # ── Optional LoRA adapter ─────────────────────────────────────────────
        if self.adapter_path:
            if not PEFT_AVAILABLE:
                raise RuntimeError(
                    "[Phi3Reasoner] adapter_path provided but peft is not installed. "
                    "Run: pip install peft"
                )
            logger.info(f"[Phi3Reasoner] Loading LoRA adapter: {self.adapter_path}")
            self._model = PeftModel.from_pretrained(
                self._model, self.adapter_path, is_trainable=False,
            )

        self._model.eval()

        # ── Post-load VRAM report ─────────────────────────────────────────────
        params = sum(p.numel() for p in self._model.parameters()) / 1e9
        vfree_after, _ = torch.cuda.mem_get_info(0)
        vram_used = vtotal - vfree_after
        logger.info(
            f"[Phi3Reasoner] ✓ Ready — {params:.2f}B params | "
            f"device=cuda (4-bit NF4) | "
            f"VRAM used={vram_used/(1024**3):.1f}GB, "
            f"free={vfree_after/(1024**3):.1f}GB"
        )

    # ── Chat template ─────────────────────────────────────────────────────────

    def _apply_chat_template(self, user_message: str, system_message: str) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_message},
                        {"role": "user",   "content": user_message},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:
                logger.warning(
                    f"[Phi3Reasoner] apply_chat_template failed ({exc}), using fallback"
                )
        return _build_phi3_prompt(user_message, system_message)

    # ── Generation ────────────────────────────────────────────────────────────

    def _generate(self, prompt: str, system_message: str, is_json_mode: bool = False) -> str:
        """
        Run one forward + generate pass on GPU.

        Tokenises in two steps:
          Step 1 — measure prompt length (no truncation)
          Step 2 — re-tokenise with max_length = ctx - safe_new_tokens
        This guarantees max_length > 0 and prevents the reshape-to-zero crash.
        """
        self._last_raw_output = ""

        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "[Phi3Reasoner] Model is not loaded. "
                "Check startup logs for loading errors."
            )

        formatted = self._apply_chat_template(prompt, system_message)

        try:
            # Step 1: measure prompt length
            probe = self._tokenizer(
                formatted,
                return_tensors="pt",
                truncation=False,
                padding=False,
                add_special_tokens=True,
            )
            prompt_token_len = probe["input_ids"].shape[-1]

            # Step 2: compute safe output budget and re-tokenise
            safe_new_tokens = self._safe_new_tokens(prompt_token_len)
            max_prompt_len  = _PHI3_CTX_LEN - safe_new_tokens

            inputs = self._tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_len,
                padding=False,
                add_special_tokens=True,
            ).to(self._model.device)

            actual_prompt_len = inputs["input_ids"].shape[-1]
            if actual_prompt_len == 0:
                logger.error(
                    "[Phi3Reasoner] Prompt tokenised to 0 tokens after truncation. "
                    f"max_prompt_len={max_prompt_len} safe_new_tokens={safe_new_tokens}."
                )
                return ""

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens":     safe_new_tokens,
                "do_sample":          self.do_sample,
                "repetition_penalty": 1.1,
                "pad_token_id":       self._tokenizer.pad_token_id,
            }
            if self._stop_token_ids:
                gen_kwargs["eos_token_id"] = self._stop_token_ids
            if self.do_sample:
                gen_kwargs["temperature"] = self.temperature
                gen_kwargs["top_p"]       = self.top_p

            if is_json_mode and STOPPING_CRITERIA_AVAILABLE:
                _stopper = _JsonCompleteStopper(self._tokenizer, owner=self)
                _stopper._prompt_len = actual_prompt_len
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_stopper])

            # Run generate() in a daemon thread with a hard wall-clock timeout
            import threading as _threading
            _output_ids_holder: List[Any] = [None]
            _gen_exc:           List[Any] = [None]

            _timeout_sec = int(os.environ.get(
                "PHI3_GENERATE_TIMEOUT",
                str(max(60, safe_new_tokens // 8 + 20))
            ))

            def _run_generate():
                try:
                    with torch.no_grad():
                        _output_ids_holder[0] = self._model.generate(
                            **inputs, **gen_kwargs
                        )
                except Exception as exc:
                    _gen_exc[0] = exc

            _gen_thread = _threading.Thread(
                target=_run_generate, daemon=True, name="phi3_generate"
            )
            _gen_thread.start()
            _gen_thread.join(timeout=_timeout_sec)

            if _gen_exc[0] is not None:
                raise _gen_exc[0]

            if _gen_thread.is_alive() or _output_ids_holder[0] is None:
                logger.warning(
                    f"[Phi3Reasoner] Generation timed out after {_timeout_sec}s "
                    f"(max_new_tokens={safe_new_tokens}). "
                    "Returning partial output from stopper if available."
                )
                return self._last_raw_output or ""

            output_ids = _output_ids_holder[0]
            decoded = self._tokenizer.decode(
                output_ids[0][actual_prompt_len:], skip_special_tokens=True
            )
            decoded = re.sub(
                r"<\|(?:end|endoftext|assistant|user|system)\|>.*$",
                "", decoded, flags=re.DOTALL,
            )
            result = decoded.strip()
            self._last_raw_output = result
            return result

        except torch.cuda.OutOfMemoryError:
            logger.error(
                "[Phi3Reasoner] CUDA out of memory during generation. "
                f"Try reducing MAX_REASONING_TOKENS (currently {self.max_new_tokens})."
            )
            torch.cuda.empty_cache()
            return ""
        except Exception as exc:
            logger.error(f"[Phi3Reasoner] Generation failed: {exc}", exc_info=True)
            return ""

    # ── Public API ────────────────────────────────────────────────────────────

    def reason(self, prompt: str) -> Any:
        """
        Run the model and parse the response as JSON.
        Returns a dict or list on success, {} on parse failure.
        Raw text is always stored in self._last_raw_output for partial recovery.
        """
        raw = self._generate(prompt, self._SYSTEM_JSON, is_json_mode=True)
        if not raw:
            return {}

        clean = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        clean = re.sub(r"\s*```\s*$", "", clean.strip())

        try:
            result = json.loads(clean)
            return result if isinstance(result, (dict, list)) else {}
        except json.JSONDecodeError:
            pass

        extracted = self._extract_balanced(clean)
        if extracted is not None:
            return extracted

        logger.warning(
            f"[Phi3Reasoner] JSON parse failed. "
            f"Raw ({len(raw)} chars): {raw[:300]}"
        )
        return {}

    @staticmethod
    def _extract_balanced(text: str) -> Any:
        """Find the first brace-balanced JSON object or array in text."""
        for open_ch, close_ch in [('{', '}'), ('[', ']')]:
            start = text.find(open_ch)
            if start == -1:
                continue
            depth   = 0
            in_str  = False
            escape  = False
            for i in range(start, len(text)):
                ch = text[i]
                if escape:
                    escape = False
                    continue
                if ch == '\\' and in_str:
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        try:
                            result = json.loads(text[start:i+1])
                            if isinstance(result, (dict, list)):
                                return result
                        except json.JSONDecodeError:
                            pass
                        break
        return None

    def reason_text(self, prompt: str) -> str:
        """Run the model and return the raw Markdown output."""
        return self._generate(prompt, self._SYSTEM_TEXT)

    def load_adapter(self, adapter_path: str) -> None:
        if not PEFT_AVAILABLE:
            raise ImportError("pip install peft")
        if self._model is None:
            raise RuntimeError("Base model not loaded.")
        logger.info(f"[Phi3Reasoner] Loading adapter: {adapter_path}")
        if isinstance(self._model, PeftModel):
            self._model.load_adapter(adapter_path, adapter_name="default")
        else:
            self._model = PeftModel.from_pretrained(
                self._model, self.adapter_path, is_trainable=False,
            )
        self._model.eval()

    @property
    def model(self):     return self._model

    @property
    def tokenizer(self): return self._tokenizer


# Drop-in alias used throughout the codebase
LlamaReasoner = Phi3Reasoner