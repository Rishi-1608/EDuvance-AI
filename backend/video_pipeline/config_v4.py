"""
video_pipeline/config.py
========================
Central configuration for the Academic Intelligence System.
All values can be overridden via environment variables.

v3 additions
------------
  slide_hist_threshold      — histogram correlation cut-off for slide detection
  slide_ssim_threshold      — SSIM score cut-off for slide detection
  slide_min_seconds         — minimum gap between accepted slide changes
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    # ── Frame extraction ──────────────────────────────────────────────────────
    # For a 3-min video: fps=1.0 → ~180 frames before slide detection.
    # For a 1-hour lecture: use fps=0.2 (1 frame/5s) → ~720 frames, slide
    # detection cuts that to ~100-200 unique slides — much more manageable.
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

    # 4-bit QLoRA — MUST be True for RTX 3050 (4 GB VRAM).
    phi3_load_in_4bit: bool = (
        os.environ.get("PHI3_LOAD_IN_4BIT", "true").lower() == "true"
    )

    # ── Whisper ───────────────────────────────────────────────────────────────
    whisper_model_size: str = os.environ.get("WHISPER_MODEL_SIZE", "base")

    # ── Storage ───────────────────────────────────────────────────────────────
    output_dir: str = os.environ.get("OUTPUT_DIR", "outputs")

    # ── Processing limits ─────────────────────────────────────────────────────
    max_frames_in_memory: int = int(os.environ.get("MAX_FRAMES_IN_MEMORY", "2000"))
    notes_sample_frames: int = int(os.environ.get("NOTES_SAMPLE_FRAMES", "15"))
    summary_sample_frames: int = int(os.environ.get("SUMMARY_SAMPLE_FRAMES", "10"))

    # ── v3: Slide change detection ────────────────────────────────────────────
    slide_hist_threshold: float = float(os.environ.get("SLIDE_HIST_THRESHOLD", "0.92"))
    slide_ssim_threshold: float = float(os.environ.get("SLIDE_SSIM_THRESHOLD", "0.85"))
    slide_min_seconds: float = float(os.environ.get("SLIDE_MIN_SECONDS", "1.0"))

    # ── v3: Context window (used to cap per-task token budgets safely) ────────
    # Phi-3-mini = 4096. Override if using a model with a larger window.
    phi3_context_length: int = int(os.environ.get("PHI3_CONTEXT_LENGTH", "4096"))

    # ── v3: Per-task token limits ─────────────────────────────────────────────
    # Raised from the default 1024 — notes and flashcards are long outputs.
    # Frame extraction and summaries are shorter, so they keep the base limit.
    max_tokens_notes:      int = int(os.environ.get("MAX_TOKENS_NOTES",      "4096"))
    max_tokens_flashcards: int = int(os.environ.get("MAX_TOKENS_FLASHCARDS", "3072"))
    max_tokens_quiz:       int = int(os.environ.get("MAX_TOKENS_QUIZ",       "2048"))
    max_tokens_summary:    int = int(os.environ.get("MAX_TOKENS_SUMMARY",    "1024"))
    # Per-frame extraction only needs a small JSON object — tight limit saves
    # ~15s per frame on Phi-3-mini (avoids generating padding tokens).
    max_tokens_frame:      int = int(os.environ.get("MAX_TOKENS_FRAME",       "400"))

    # ── v3: OCR noise filter ──────────────────────────────────────────────────
    # Minimum number of *word characters* (not raw chars) in OCR output before
    # the LLM is invoked.  Raises the bar above noise strings like "nnD", "0p8".
    min_ocr_word_chars: int = int(os.environ.get("MIN_OCR_WORD_CHARS", "15"))


# Singleton
config = Config()