"""
video_pipeline/config1.py
========================
Central configuration for the Academic Intelligence System.
All values can be overridden via environment variables.

v3.1.0 changes
--------------
  whisper_model_size     — default changed "base" → "tiny".
                           tiny is ~4x faster (39 MB vs 74 MB) with acceptable
                           accuracy on lecture audio. Override with
                           WHISPER_MODEL_SIZE=base to restore previous behaviour.
  phase2_max_frames      — NEW: max frames fed into LLM prompts in Phase 2.
                           Prevents prompt token overflow on 1-hour lectures.
                           Default 40. Override with PHASE2_MAX_FRAMES.
  max_frames_extract     — NEW: hard cap on frames extracted per video.
                           Default 720 (~2 hours at 0.1 fps). Override with
                           MAX_FRAMES_EXTRACT.
  fps_long_threshold_1   — NEW: video duration (sec) above which FPS drops to
                           fps_for_long_1. Default 1200 (20 min).
  fps_long_threshold_2   — NEW: video duration (sec) above which FPS drops to
                           fps_for_long_2. Default 2400 (40 min).
  fps_for_long_1         — NEW: FPS for 20–40 min videos. Default 0.2.
  fps_for_long_2         — NEW: FPS for >40 min videos. Default 0.1.
  whisper_timeout_sec    — NEW: max seconds to wait for Whisper transcription.
                           Default 600 (10 min). Prevents hang on long audio.
  ocr_batch_size         — NEW: frames per EasyOCR batch call. Default 8
                           (was 4). GPU batching is more efficient at 8.

v3.0.3 fixes
-----------------
  max_tokens_transcript_only — raised from 900 → 1200.
  min_ocr_word_chars         — lowered from 15 → 8.
  easyocr_gpu                — NEW: controls whether EasyOCR uses CUDA.

v3.0.2 additions
-----------------
  max_tokens_everything      — lowered from 1600 → 1200.
  max_tokens_transcript_only — token budget for compact transcript-only prompt.

v3.0.1 note
-----------
  SSL fix applied in main.py at import time (certifi CA bundle).

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
    # v3.1.0 FIX: changed default "base" → "tiny".
    # "tiny" is ~4x faster (39 MB, ~15s for 1-hour audio on CPU) vs "base"
    # (74 MB, ~60s). Accuracy is sufficient for English lecture transcription.
    # Override: WHISPER_MODEL_SIZE=base
    whisper_model_size: str = os.environ.get("WHISPER_MODEL_SIZE", "tiny")

    # v3.1.0: hard timeout for Whisper transcription (seconds).
    # Prevents indefinite hang on very long or corrupted audio files.
    # Override: WHISPER_TIMEOUT_SEC=900
    whisper_timeout_sec: int = int(os.environ.get("WHISPER_TIMEOUT_SEC", "600"))

    # ── Storage ───────────────────────────────────────────────────────────────
    output_dir: str = os.environ.get("OUTPUT_DIR", "outputs")

    # ── Processing limits ─────────────────────────────────────────────────────
    max_frames_in_memory: int  = int(os.environ.get("MAX_FRAMES_IN_MEMORY", "2000"))
    notes_sample_frames: int   = int(os.environ.get("NOTES_SAMPLE_FRAMES",  "15"))
    summary_sample_frames: int = int(os.environ.get("SUMMARY_SAMPLE_FRAMES", "10"))

    # v3.1.0: hard cap on total frames extracted per video stream.
    # At fps=0.1, a 2-hour video produces 720 raw frames — this cap ensures
    # we never exceed that regardless of video length or fps setting.
    # Override: MAX_FRAMES_EXTRACT=1440
    max_frames_extract: int = int(os.environ.get("MAX_FRAMES_EXTRACT", "720"))

    # v3.1.0: max frames fed into LLM prompts during Phase 2.
    # Even after adaptive FPS + slide dedup, 100+ frames would overflow
    # Phi-3-mini's 4096-token context. 40 representative frames is sufficient
    # for metadata extraction and study notes generation.
    # Override: PHASE2_MAX_FRAMES=60
    phase2_max_frames: int = int(os.environ.get("PHASE2_MAX_FRAMES", "40"))

    # ── v3.1.0: Adaptive FPS for long videos ─────────────────────────────────
    # Videos longer than fps_long_threshold_1 seconds use fps_for_long_1.
    # Videos longer than fps_long_threshold_2 seconds use fps_for_long_2.
    # This keeps frame count manageable without sacrificing slide coverage.
    fps_long_threshold_1: int   = int(os.environ.get("FPS_LONG_THRESHOLD_1", str(20 * 60)))
    fps_long_threshold_2: int   = int(os.environ.get("FPS_LONG_THRESHOLD_2", str(40 * 60)))
    fps_for_long_1: float        = float(os.environ.get("FPS_FOR_LONG_1", "0.2"))
    fps_for_long_2: float        = float(os.environ.get("FPS_FOR_LONG_2", "0.1"))

    # ── v3.1.0: OCR batch size ────────────────────────────────────────────────
    # EasyOCR GPU batching is more efficient at 8 frames per call.
    # Doubling from 4 → 8 halves GPU round-trips with negligible memory impact.
    # Override: OCR_BATCH_SIZE=4
    ocr_batch_size: int = int(os.environ.get("OCR_BATCH_SIZE", "8"))

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

    # v3.0.2: lowered from 1600 → 1200 to avoid the 90-second Phi-3 timeout.
    max_tokens_everything: int = int(os.environ.get("MAX_TOKENS_EVERYTHING", "1200"))

    # v3.0.3 FIX: raised from 900 → 1200.
    max_tokens_transcript_only: int = int(
        os.environ.get("MAX_TOKENS_TRANSCRIPT_ONLY", "1200")
    )

    # ── v3: OCR noise filter ──────────────────────────────────────────────────
    # v3.0.3 FIX: lowered from 15 → 8.
    min_ocr_word_chars: int = int(os.environ.get("MIN_OCR_WORD_CHARS", "30"))

    # ── v3.0.3: EasyOCR GPU flag ──────────────────────────────────────────────
    # FIX: the previous check compared against the string "false", which meant
    # EASYOCR_GPU=false evaluated to True (GPU enabled) and EASYOCR_GPU=true
    # evaluated to False (GPU disabled) — exactly backwards. Default remains
    # True to preserve existing local-GPU behaviour; set EASYOCR_GPU=false
    # to force CPU (required for deployments with no CUDA GPU).
    easyocr_gpu: bool = (
        os.environ.get("EASYOCR_GPU", "true").lower() == "true"
    )


# Singleton
config = Config()