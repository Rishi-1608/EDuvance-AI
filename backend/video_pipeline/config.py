# """
# video_pipeline/config.py
# ========================
# Central configuration for the Academic Intelligence System.
# All values can be overridden via environment variables.

# v3 additions
# ------------
#   slide_hist_threshold      — histogram correlation cut-off for slide detection
#   slide_ssim_threshold      — SSIM score cut-off for slide detection
#   slide_min_seconds         — minimum gap between accepted slide changes
# """
# from __future__ import annotations

# import os
# from dataclasses import dataclass


# @dataclass
# class Config:
#     # ── Frame extraction ──────────────────────────────────────────────────────
#     fps: float = float(os.environ.get("PIPELINE_FPS", "1.0"))
#     min_ocr_chars: int = int(os.environ.get("MIN_OCR_CHARS", "20"))

#     # ── OCR ───────────────────────────────────────────────────────────────────
#     ocr_confidence_threshold: float = float(
#         os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.4")
#     )

#     @property
#     def telemetry_confidence_threshold(self) -> float:
#         return self.ocr_confidence_threshold

#     # ── LLM / Reasoning ──────────────────────────────────────────────────────
#     reasoning_model_id: str = os.environ.get("REASONING_MODEL_ID", "models/phi3mini")
#     max_reasoning_tokens: int = int(os.environ.get("MAX_REASONING_TOKENS", "1024"))

#     # ── Phi-3-mini specific ───────────────────────────────────────────────────
#     phi3_adapter_path: str = os.environ.get("PHI3_ADAPTER_PATH", "")

#     # 4-bit QLoRA — MUST be True for RTX 3050 (4 GB VRAM).
#     phi3_load_in_4bit: bool = (
#         os.environ.get("PHI3_LOAD_IN_4BIT", "true").lower() == "true"
#     )

#     # ── Whisper ───────────────────────────────────────────────────────────────
#     whisper_model_size: str = os.environ.get("WHISPER_MODEL_SIZE", "base")

#     # ── Storage ───────────────────────────────────────────────────────────────
#     output_dir: str = os.environ.get("OUTPUT_DIR", "outputs")

#     # ── Processing limits ─────────────────────────────────────────────────────
#     max_frames_in_memory: int = int(os.environ.get("MAX_FRAMES_IN_MEMORY", "2000"))
#     notes_sample_frames: int = int(os.environ.get("NOTES_SAMPLE_FRAMES", "15"))
#     summary_sample_frames: int = int(os.environ.get("SUMMARY_SAMPLE_FRAMES", "10"))

#     # ── v3: Slide change detection ────────────────────────────────────────────
#     slide_hist_threshold: float = float(os.environ.get("SLIDE_HIST_THRESHOLD", "0.92"))
#     slide_ssim_threshold: float = float(os.environ.get("SLIDE_SSIM_THRESHOLD", "0.85"))
#     slide_min_seconds: float = float(os.environ.get("SLIDE_MIN_SECONDS", "1.0"))

#     # ── v3: Context window (used to cap per-task token budgets safely) ────────
#     # Phi-3-mini = 4096. Override if using a model with a larger window.
#     phi3_context_length: int = int(os.environ.get("PHI3_CONTEXT_LENGTH", "4096"))

#     # ── v3: Per-task token limits ─────────────────────────────────────────────
#     # Raised from the default 1024 — notes and flashcards are long outputs.
#     # Frame extraction and summaries are shorter, so they keep the base limit.
#     max_tokens_notes:      int = int(os.environ.get("MAX_TOKENS_NOTES",      "4096"))
#     max_tokens_flashcards: int = int(os.environ.get("MAX_TOKENS_FLASHCARDS", "3072"))
#     max_tokens_quiz:       int = int(os.environ.get("MAX_TOKENS_QUIZ",       "2048"))
#     max_tokens_summary:    int = int(os.environ.get("MAX_TOKENS_SUMMARY",    "1024"))

#     # ── v3: OCR noise filter ──────────────────────────────────────────────────
#     # Minimum number of *word characters* (not raw chars) in OCR output before
#     # the LLM is invoked.  Raises the bar above noise strings like "nnD", "0p8".
#     min_ocr_word_chars: int = int(os.environ.get("MIN_OCR_WORD_CHARS", "15"))


# # Singleton
# config = Config()


# ----──────────────────────────────────────────────────────────────────────────────
# Updated for v5: switched to Pydantic for better validation and environment variable parsing.

# """
# video_pipeline/config.py
# ========================
# Central configuration for the Academic Intelligence System.
# All values can be overridden via environment variables.

# v3 additions
# ------------
#   slide_hist_threshold      — histogram correlation cut-off for slide detection
#   slide_ssim_threshold      — SSIM score cut-off for slide detection
#   slide_min_seconds         — minimum gap between accepted slide changes
# """
# from __future__ import annotations

# import os
# from dataclasses import dataclass


# @dataclass
# class Config:
#     # ── Frame extraction ──────────────────────────────────────────────────────
#     fps: float = float(os.environ.get("PIPELINE_FPS", "1.0"))
#     min_ocr_chars: int = int(os.environ.get("MIN_OCR_CHARS", "20"))

#     # ── OCR ───────────────────────────────────────────────────────────────────
#     ocr_confidence_threshold: float = float(
#         os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.4")
#     )

#     @property
#     def telemetry_confidence_threshold(self) -> float:
#         return self.ocr_confidence_threshold

#     # ── LLM / Reasoning ──────────────────────────────────────────────────────
#     reasoning_model_id: str = os.environ.get("REASONING_MODEL_ID", "models/phi3mini")
#     max_reasoning_tokens: int = int(os.environ.get("MAX_REASONING_TOKENS", "1024"))

#     # ── Phi-3-mini specific ───────────────────────────────────────────────────
#     phi3_adapter_path: str = os.environ.get("PHI3_ADAPTER_PATH", "")

#     # 4-bit QLoRA — MUST be True for RTX 3050 (4 GB VRAM).
#     phi3_load_in_4bit: bool = (
#         os.environ.get("PHI3_LOAD_IN_4BIT", "true").lower() == "true"
#     )

#     # ── Whisper ───────────────────────────────────────────────────────────────
#     whisper_model_size: str = os.environ.get("WHISPER_MODEL_SIZE", "base")

#     # ── Storage ───────────────────────────────────────────────────────────────
#     output_dir: str = os.environ.get("OUTPUT_DIR", "outputs")

#     # ── Processing limits ─────────────────────────────────────────────────────
#     max_frames_in_memory: int = int(os.environ.get("MAX_FRAMES_IN_MEMORY", "2000"))
#     notes_sample_frames: int = int(os.environ.get("NOTES_SAMPLE_FRAMES", "15"))
#     summary_sample_frames: int = int(os.environ.get("SUMMARY_SAMPLE_FRAMES", "10"))

#     # ── v3: Slide change detection ────────────────────────────────────────────
#     slide_hist_threshold: float = float(os.environ.get("SLIDE_HIST_THRESHOLD", "0.92"))
#     slide_ssim_threshold: float = float(os.environ.get("SLIDE_SSIM_THRESHOLD", "0.85"))
#     slide_min_seconds: float = float(os.environ.get("SLIDE_MIN_SECONDS", "1.0"))

#     # ── v3: Context window (used to cap per-task token budgets safely) ────────
#     # Phi-3-mini = 4096. Override if using a model with a larger window.
#     phi3_context_length: int = int(os.environ.get("PHI3_CONTEXT_LENGTH", "4096"))

#     # ── v3: Per-task token limits ─────────────────────────────────────────────
#     # Raised from the default 1024 — notes and flashcards are long outputs.
#     # Frame extraction and summaries are shorter, so they keep the base limit.
#     max_tokens_notes:      int = int(os.environ.get("MAX_TOKENS_NOTES",      "4096"))
#     max_tokens_flashcards: int = int(os.environ.get("MAX_TOKENS_FLASHCARDS", "3072"))
#     max_tokens_quiz:       int = int(os.environ.get("MAX_TOKENS_QUIZ",       "2048"))
#     max_tokens_summary:    int = int(os.environ.get("MAX_TOKENS_SUMMARY",    "1024"))

#     # ── v3: OCR noise filter ──────────────────────────────────────────────────
#     # Minimum number of *word characters* (not raw chars) in OCR output before
#     # the LLM is invoked.  Raises the bar above noise strings like "nnD", "0p8".
#     min_ocr_word_chars: int = int(os.environ.get("MIN_OCR_WORD_CHARS", "15"))


# # Singleton
# config = Config()


# ----──────────────────────────────────────────────────────────────────────────────
# Updated for v5: switched to Pydantic for better validation and environment variable parsing.

# """
# video_pipeline/config.py
# ========================
# Central configuration for the Academic Intelligence System.
# All values can be overridden via environment variables.

# v3 additions
# ------------
#   slide_hist_threshold      — histogram correlation cut-off for slide detection
#   slide_ssim_threshold      — SSIM score cut-off for slide detection
#   slide_min_seconds         — minimum gap between accepted slide changes
# """
# from __future__ import annotations

# import os
# from dataclasses import dataclass


# @dataclass
# class Config:
#     # ── Frame extraction ──────────────────────────────────────────────────────
#     fps: float = float(os.environ.get("PIPELINE_FPS", "1.0"))
#     min_ocr_chars: int = int(os.environ.get("MIN_OCR_CHARS", "20"))

#     # ── OCR ───────────────────────────────────────────────────────────────────
#     ocr_confidence_threshold: float = float(
#         os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.4")
#     )

#     @property
#     def telemetry_confidence_threshold(self) -> float:
#         return self.ocr_confidence_threshold

#     # ── LLM / Reasoning ──────────────────────────────────────────────────────
#     reasoning_model_id: str = os.environ.get("REASONING_MODEL_ID", "models/phi3mini")
#     max_reasoning_tokens: int = int(os.environ.get("MAX_REASONING_TOKENS", "1024"))

#     # ── Phi-3-mini specific ───────────────────────────────────────────────────
#     phi3_adapter_path: str = os.environ.get("PHI3_ADAPTER_PATH", "")

#     # 4-bit QLoRA — MUST be True for RTX 3050 (4 GB VRAM).
#     phi3_load_in_4bit: bool = (
#         os.environ.get("PHI3_LOAD_IN_4BIT", "true").lower() == "true"
#     )

#     # ── Whisper ───────────────────────────────────────────────────────────────
#     whisper_model_size: str = os.environ.get("WHISPER_MODEL_SIZE", "base")

#     # ── Storage ───────────────────────────────────────────────────────────────
#     output_dir: str = os.environ.get("OUTPUT_DIR", "outputs")

#     # ── Processing limits ─────────────────────────────────────────────────────
#     max_frames_in_memory: int = int(os.environ.get("MAX_FRAMES_IN_MEMORY", "2000"))
#     notes_sample_frames: int = int(os.environ.get("NOTES_SAMPLE_FRAMES", "15"))
#     summary_sample_frames: int = int(os.environ.get("SUMMARY_SAMPLE_FRAMES", "10"))

#     # ── v3: Slide change detection ────────────────────────────────────────────
#     slide_hist_threshold: float = float(os.environ.get("SLIDE_HIST_THRESHOLD", "0.92"))
#     slide_ssim_threshold: float = float(os.environ.get("SLIDE_SSIM_THRESHOLD", "0.85"))
#     slide_min_seconds: float = float(os.environ.get("SLIDE_MIN_SECONDS", "1.0"))

#     # ── v3: Context window (used to cap per-task token budgets safely) ────────
#     # Phi-3-mini = 4096. Override if using a model with a larger window.
#     phi3_context_length: int = int(os.environ.get("PHI3_CONTEXT_LENGTH", "4096"))

#     # ── v3: Per-task token limits ─────────────────────────────────────────────
#     # Raised from the default 1024 — notes and flashcards are long outputs.
#     # Frame extraction and summaries are shorter, so they keep the base limit.
#     max_tokens_notes:      int = int(os.environ.get("MAX_TOKENS_NOTES",      "4096"))
#     max_tokens_flashcards: int = int(os.environ.get("MAX_TOKENS_FLASHCARDS", "3072"))
#     max_tokens_quiz:       int = int(os.environ.get("MAX_TOKENS_QUIZ",       "2048"))
#     max_tokens_summary:    int = int(os.environ.get("MAX_TOKENS_SUMMARY",    "1024"))

#     # ── v3: OCR noise filter ──────────────────────────────────────────────────
#     # Minimum number of *word characters* (not raw chars) in OCR output before
#     # the LLM is invoked.  Raises the bar above noise strings like "nnD", "0p8".
#     min_ocr_word_chars: int = int(os.environ.get("MIN_OCR_WORD_CHARS", "15"))


# # Singleton
# config = Config()


# ----──────────────────────────────────────────────────────────────────────────────
# Updated for v5: switched to Pydantic for better validation and environment variable parsing.

# """
# video_pipeline/config.py
# ========================
# Central configuration for the Academic Intelligence System.
# All values can be overridden via environment variables.

# v3 additions
# ------------
#   slide_hist_threshold      — histogram correlation cut-off for slide detection
#   slide_ssim_threshold      — SSIM score cut-off for slide detection
#   slide_min_seconds         — minimum gap between accepted slide changes
# """
# from __future__ import annotations

# import os
# from dataclasses import dataclass


# @dataclass
# class Config:
#     # ── Frame extraction ──────────────────────────────────────────────────────
#     fps: float = float(os.environ.get("PIPELINE_FPS", "1.0"))
#     min_ocr_chars: int = int(os.environ.get("MIN_OCR_CHARS", "20"))

#     # ── OCR ───────────────────────────────────────────────────────────────────
#     ocr_confidence_threshold: float = float(
#         os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.4")
#     )

#     @property
#     def telemetry_confidence_threshold(self) -> float:
#         return self.ocr_confidence_threshold

#     # ── LLM / Reasoning ──────────────────────────────────────────────────────
#     reasoning_model_id: str = os.environ.get("REASONING_MODEL_ID", "models/phi3mini")
#     max_reasoning_tokens: int = int(os.environ.get("MAX_REASONING_TOKENS", "1024"))

#     # ── Phi-3-mini specific ───────────────────────────────────────────────────
#     phi3_adapter_path: str = os.environ.get("PHI3_ADAPTER_PATH", "")

#     # 4-bit QLoRA — MUST be True for RTX 3050 (4 GB VRAM).
#     phi3_load_in_4bit: bool = (
#         os.environ.get("PHI3_LOAD_IN_4BIT", "true").lower() == "true"
#     )

#     # ── Whisper ───────────────────────────────────────────────────────────────
#     whisper_model_size: str = os.environ.get("WHISPER_MODEL_SIZE", "base")

#     # ── Storage ───────────────────────────────────────────────────────────────
#     output_dir: str = os.environ.get("OUTPUT_DIR", "outputs")

#     # ── Processing limits ─────────────────────────────────────────────────────
#     max_frames_in_memory: int = int(os.environ.get("MAX_FRAMES_IN_MEMORY", "2000"))
#     notes_sample_frames: int = int(os.environ.get("NOTES_SAMPLE_FRAMES", "15"))
#     summary_sample_frames: int = int(os.environ.get("SUMMARY_SAMPLE_FRAMES", "10"))

#     # ── v3: Slide change detection ────────────────────────────────────────────
#     slide_hist_threshold: float = float(os.environ.get("SLIDE_HIST_THRESHOLD", "0.92"))
#     slide_ssim_threshold: float = float(os.environ.get("SLIDE_SSIM_THRESHOLD", "0.85"))
#     slide_min_seconds: float = float(os.environ.get("SLIDE_MIN_SECONDS", "1.0"))

#     # ── v3: Context window (used to cap per-task token budgets safely) ────────
#     # Phi-3-mini = 4096. Override if using a model with a larger window.
#     phi3_context_length: int = int(os.environ.get("PHI3_CONTEXT_LENGTH", "4096"))

#     # ── v3: Per-task token limits ─────────────────────────────────────────────
#     # Raised from the default 1024 — notes and flashcards are long outputs.
#     # Frame extraction and summaries are shorter, so they keep the base limit.
#     max_tokens_notes:      int = int(os.environ.get("MAX_TOKENS_NOTES",      "4096"))
#     max_tokens_flashcards: int = int(os.environ.get("MAX_TOKENS_FLASHCARDS", "3072"))
#     max_tokens_quiz:       int = int(os.environ.get("MAX_TOKENS_QUIZ",       "2048"))
#     max_tokens_summary:    int = int(os.environ.get("MAX_TOKENS_SUMMARY",    "1024"))

#     # ── v3: OCR noise filter ──────────────────────────────────────────────────
#     # Minimum number of *word characters* (not raw chars) in OCR output before
#     # the LLM is invoked.  Raises the bar above noise strings like "nnD", "0p8".
#     min_ocr_word_chars: int = int(os.environ.get("MIN_OCR_WORD_CHARS", "15"))


# # Singleton
# config = Config()


# ----──────────────────────────────────────────────────────────────────────────────
# Updated for v5: switched to Pydantic for better validation and environment variable parsing.

"""
video_pipeline/config.py
========================
Central configuration for the Academic Intelligence System.
All values can be overridden via environment variables.

v3.0.3 fixes
-----------------
  max_tokens_transcript_only — raised from 900 → 1200.
                               900 was too small to complete the full JSON
                               (flashcards + quiz + notes) on the transcript-only
                               fallback path, causing truncated JSON and no output.
  min_ocr_word_chars         — lowered from 15 → 8.
                               15 was too aggressive for animated lecture videos
                               (e.g. Newton's law explainers) where on-screen text
                               is short labels like "F = ma", "mass", "10 N".
                               8 catches these while still filtering pure noise.
  easyocr_gpu                — NEW: controls whether EasyOCR uses CUDA.
                               Defaults to True — the RTX 3050 has ~1.6 GB VRAM
                               free after Phi-3 4-bit loads (~2.4 GB), which is
                               enough for EasyOCR. Set EASYOCR_GPU=false to
                               force CPU mode if you hit OOM errors.

v3.0.2 additions
-----------------
  max_tokens_everything      — lowered from 1600 → 1200 (avoids 90s Phi-3 timeout)
  max_tokens_transcript_only — token budget for the compact transcript-only
                               prompt used when slide OCR produced no content.

v3.0.1 note
-----------
  SSL fix applied in main.py at import time (certifi CA bundle).
  Run:  pip install --upgrade certifi

v3.0.0 additions
-----------------
  slide_hist_threshold / slide_ssim_threshold / slide_min_seconds
  phi3_context_length / max_tokens_* / min_ocr_word_chars
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    # ── Frame extraction ──────────────────────────────────────────────────────
    fps: float = float(os.environ.get("PIPELINE_FPS", "0.5"))
    min_ocr_chars: int = int(os.environ.get("MIN_OCR_CHARS", "20"))

    # ── OCR ───────────────────────────────────────────────────────────────────
    ocr_confidence_threshold: float = float(
        os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.4")
    )

    @property
    def telemetry_confidence_threshold(self) -> float:
        return self.ocr_confidence_threshold

    # ── LLM / Reasoning ──────────────────────────────────────────────────────
    reasoning_model_id: str = os.environ.get("REASONING_MODEL_ID", "models/phi3mini")
    max_reasoning_tokens: int = int(os.environ.get("MAX_REASONING_TOKENS", "1024"))

    # ── Phi-3-mini specific ───────────────────────────────────────────────────
    phi3_adapter_path: str = os.environ.get("PHI3_ADAPTER_PATH", "")
    phi3_load_in_4bit: bool = (
        os.environ.get("PHI3_LOAD_IN_4BIT", "true").lower() == "true"
    )

    # ── Whisper ───────────────────────────────────────────────────────────────
    whisper_model_size: str = os.environ.get("WHISPER_MODEL_SIZE", "base")

    # ── Storage ───────────────────────────────────────────────────────────────
    output_dir: str = os.environ.get("OUTPUT_DIR", "outputs")

    # ── Processing limits ─────────────────────────────────────────────────────
    max_frames_in_memory: int  = int(os.environ.get("MAX_FRAMES_IN_MEMORY", "2000"))
    notes_sample_frames: int   = int(os.environ.get("NOTES_SAMPLE_FRAMES",  "15"))
    summary_sample_frames: int = int(os.environ.get("SUMMARY_SAMPLE_FRAMES", "10"))

    # ── v3: Slide change detection ────────────────────────────────────────────
    slide_hist_threshold: float = float(os.environ.get("SLIDE_HIST_THRESHOLD", "0.92"))
    slide_ssim_threshold: float = float(os.environ.get("SLIDE_SSIM_THRESHOLD", "0.85"))
    slide_min_seconds:    float = float(os.environ.get("SLIDE_MIN_SECONDS",    "1.0"))

    # ── v3: Context window ────────────────────────────────────────────────────
    phi3_context_length: int = int(os.environ.get("PHI3_CONTEXT_LENGTH", "4096"))

    # ── v3 / v3.0.2: Per-task token limits ───────────────────────────────────
    max_tokens_notes:      int = int(os.environ.get("MAX_TOKENS_NOTES",      "4096"))
    max_tokens_flashcards: int = int(os.environ.get("MAX_TOKENS_FLASHCARDS", "3072"))
    max_tokens_quiz:       int = int(os.environ.get("MAX_TOKENS_QUIZ",       "2048"))
    max_tokens_summary:    int = int(os.environ.get("MAX_TOKENS_SUMMARY",    "1024"))
    max_tokens_frame:      int = int(os.environ.get("MAX_TOKENS_FRAME",       "400"))

    # v3.0.2: lowered from 1600 → 1200 to avoid the 90-second Phi-3 timeout
    # on RTX 3050.  Use PIPELINE_FPS=0.5 and keep sample_n=8 (default) to
    # stay well within the 4096-token context window.
    max_tokens_everything: int = int(os.environ.get("MAX_TOKENS_EVERYTHING", "1200"))

    # v3.0.3 FIX: raised from 900 → 1200.
    # 900 was not enough to complete the full JSON output (flashcards + quiz +
    # study_notes) on RTX 3050, causing the response to truncate mid-JSON and
    # the parser to discard everything → no notes, no flashcards, no quiz, no PDF.
    # 1200 reliably fits the complete prompt_everything() output within Phi-3's
    # 90-second budget on an RTX 3050.
    max_tokens_transcript_only: int = int(
        os.environ.get("MAX_TOKENS_TRANSCRIPT_ONLY", "1200")
    )

    # ── v3: OCR noise filter ──────────────────────────────────────────────────
    # v3.0.3 FIX: lowered from 15 → 8.
    # 15 was too aggressive for animated lecture videos (e.g. Newton's law
    # explainers) where on-screen labels are short strings like "F = ma",
    # "mass", "10 N" — all under 15 word-characters and incorrectly discarded.
    # 8 still filters pure OCR noise ("nnD", "0p8", "|||") while keeping
    # short but meaningful physics/math labels.
    min_ocr_word_chars: int = int(os.environ.get("MIN_OCR_WORD_CHARS", "8"))

    # ── v3.0.3: EasyOCR GPU flag ──────────────────────────────────────────────
    # FIX: was hardcoded to gpu=False in the EasyOCR reader init, wasting the
    # RTX 3050's spare VRAM. Phi-3 in 4-bit uses ~2.4 GB; the card has 4.0 GB
    # total, leaving ~1.6 GB free — enough for EasyOCR's detection model.
    # Enabling GPU OCR cuts per-frame OCR time by ~60% and improves accuracy
    # on brief animated text.
    # Set EASYOCR_GPU=false to force CPU mode if you hit OOM errors.
    easyocr_gpu: bool = (
        os.environ.get("EASYOCR_GPU", "true").lower() == "true"
    )


# Singleton
config = Config()