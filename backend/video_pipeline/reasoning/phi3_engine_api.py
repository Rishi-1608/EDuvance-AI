"""
video_pipeline/reasoning/phi3_engine_api.py
==========================================
Drop-in replacement for Phi3Reasoner (phi3_engine.py) that calls
Phi-3-mini-4k-instruct through Hugging Face's Inference API instead of
loading the model locally on a GPU.

Use this when deploying to a host with no CUDA GPU (e.g. a free
Hugging Face Space on CPU Basic). Nothing else in the codebase needs to
change — same public interface as the local Phi3Reasoner:

  reason(prompt)      -> dict | list
  reason_text(prompt) -> str
  context_length       (property)
  model / tokenizer     (properties, return None here — no local model)
  load_adapter(path)    (raises NotImplementedError — no LoRA over the API)

Requires:
  pip install huggingface_hub
  env var HF_TOKEN set to a Hugging Face access token
    (create one at https://huggingface.co/settings/tokens — "read" scope is enough)

Optional env vars:
  PHI3_HF_MODEL_ID       — override the model repo id (default below)
  PHI3_CONTEXT_LENGTH    — same meaning as the local engine (default 4096)
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning(
        "[Phi3Reasoner/API] huggingface_hub not installed. "
        "Run: pip install huggingface_hub"
    )

_PHI3_CTX_LEN = int(os.environ.get("PHI3_CONTEXT_LENGTH", "4096"))
_HF_MODEL_ID = os.environ.get("PHI3_HF_MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")


class Phi3Reasoner:
    """
    Wraps Phi-3-mini-4k-instruct via the Hugging Face Inference API.

    Drop-in replacement for the local-GPU Phi3Reasoner in phi3_engine.py:
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
        model_id:       str           = "models/phi3mini",  # kept for signature compat, unused
        max_new_tokens: int           = 1024,
        device:         str           = "cpu",   # accepted, unused — no local device here
        load_in_4bit:   bool          = True,    # accepted, unused — API handles serving
        temperature:    float         = 0.1,
        top_p:          float         = 0.95,
        do_sample:      bool          = False,
        adapter_path:   Optional[str] = None,
    ) -> None:
        if not HF_HUB_AVAILABLE:
            raise RuntimeError(
                "[Phi3Reasoner/API] huggingface_hub not installed. "
                "Run: pip install huggingface_hub"
            )

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "[Phi3Reasoner/API] HF_TOKEN environment variable not set. "
                "Create a token at https://huggingface.co/settings/tokens "
                "and set it as a secret in your deployment environment."
            )

        self.model_id       = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.top_p          = top_p
        self.do_sample      = do_sample
        self.adapter_path   = adapter_path
        self._last_raw_output: str = ""

        self._client = InferenceClient(model=_HF_MODEL_ID, token=hf_token)

        if adapter_path:
            logger.warning(
                "[Phi3Reasoner/API] adapter_path=%s was provided, but LoRA adapters "
                "are not supported through the Hugging Face Inference API. "
                "Ignoring adapter_path — serving the base model only.",
                adapter_path,
            )

        logger.info(
            "[Phi3Reasoner/API] Ready — using HF Inference API for model '%s'.",
            _HF_MODEL_ID,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def context_length(self) -> int:
        return _PHI3_CTX_LEN

    # ── Generation ────────────────────────────────────────────────────────────

    def _generate(self, prompt: str, system_message: str, is_json_mode: bool = False) -> str:
        self._last_raw_output = ""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": prompt},
        ]

        try:
            response = self._client.chat_completion(
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else 0.01,
                top_p=self.top_p,
            )
            result = (response.choices[0].message.content or "").strip()
            self._last_raw_output = result
            return result
        except Exception as exc:
            logger.error(f"[Phi3Reasoner/API] Generation failed: {exc}", exc_info=True)
            return ""

    # ── Public API ────────────────────────────────────────────────────────────

    def reason(self, prompt: str) -> Any:
        """
        Run the model and parse the response as JSON.
        Returns a dict or list on success, {} on parse failure.
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
            f"[Phi3Reasoner/API] JSON parse failed. "
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
            depth  = 0
            in_str = False
            escape = False
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
                            result = json.loads(text[start:i + 1])
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
        raise NotImplementedError(
            "[Phi3Reasoner/API] LoRA adapters are not supported when using the "
            "Hugging Face Inference API. Use the local phi3_engine.Phi3Reasoner "
            "on a GPU host if you need adapter support."
        )

    @property
    def model(self):
        return None

    @property
    def tokenizer(self):
        return None


# Drop-in alias used throughout the codebase
LlamaReasoner = Phi3Reasoner
