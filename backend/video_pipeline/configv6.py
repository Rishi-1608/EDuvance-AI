"""
video_pipeline/config1.py
========================
Central configuration for the Academic Intelligence System.
All values can be overridden via environment variables.

v3.3.0 changes
--------------
  paddleocr_available    — NEW: detected at runtime; PaddleOCR preferred over EasyOCR
                           when installed (2-3× faster for Latin/English text).
                           Install: pip install paddlepaddle paddleocr

  imagehash_phash        — NEW: perceptual hash slide dedup replaces SSIM.
                           ~10× faster; works on 64×36 thumbnails.
                           Install: pip install imagehash Pillow
                           Override threshold: PHASH_THRESHOLD=8

  slide_thumb_w/h        — NEW: thumbnail size for slide comparison.
                           Default 64×36. Override: SLIDE_THUMB_W, SLIDE_THUMB_H.

  ocr_batch_size         — Raised default 8 → 16. GPU round-trips halved.

  ocr_snippet_chars      — NEW: max OCR characters per frame fed into LLM prompts.
                           Prevents token overflow without losing topic signal.
                           Default 80. Override: OCR_SNIPPET_CHARS.

  phi3_single_call       — NEW: merge metadata + notes into one LLM call.
                           Saves round-trip latency on fast local models.
                           Default false. Set PHI3_SINGLE_CALL=true to enable.

  max_tokens_single_call — NEW: token budget for merged single-call Phase 2.
                           Default 1000. Override: PHI3_MAX_TOKENS_SINGLE_CALL.

  index_write_interval   — NEW: frame index flush frequency (frames).
                           Was 10, now 50 — reduces disk writes in hot path.
                           Override: INDEX_WRITE_INTERVAL.

  ramdisk_path           — NEW: /dev/shm path for hot-path JPEG writes.
                           Auto-detected; no config needed on Linux.

  adaptive_fps_bug       — FIXED: _adaptive_fps() now correctly returns
                           _FPS_FOR_LONG_2 for videos > fps_long_threshold_2.
                           Previously returned `_FPS_LONG_THRESHOLD_2 and
                           _FPS_FOR_LONG_2` which always resolved to
                           _FPS_FOR_LONG_2 but only by Python truthy accident.

  whisper_cpu_threads    — Changed default "4" → os.cpu_count().
                           Uses all available cores for CPU transcription.

v3.2.0 changes
--------------
  whisper_faster_whisper — faster-whisper CTranslate2 primary backend.
  whisper_openai_fallback— openai-whisper chunked fallback.
  chunk_timeout_sec      — per-segment timeout for faster-whisper generator.
  total_timeout_sec      — wall-clock timeout for full transcription.

v3.1.0 changes
--------------
  whisper_model_size     — default "base" → "tiny".
  phase2_max_frames      — max frames fed into LLM prompts.
  max_frames_extract     — hard cap on frames extracted per video.
  fps_long_threshold_1/2 — adaptive FPS duration thresholds.
  fps_for_long_1/2       — FPS values for long videos.
  whisper_timeout_sec    — max Whisper wall-clock time.
  ocr_batch_size         — frames per EasyOCR/PaddleOCR batch call.

v3.0.3 fixes
------------
  max_tokens_transcript_only — raised 900 → 1200.
  min_ocr_word_chars         — lowered 15 → 8 → 30.
  easyocr_gpu                — controls EasyOCR CUDA usage.

Quick reference — env vars added in v3.3.0
------------------------------------------
  PADDLEOCR_DISABLE=true     Force EasyOCR even if PaddleOCR is installed.
  IMAGEHASH_DISABLE=true     Force histogram slide detection.
  PHASH_THRESHOLD=8          Hamming distance threshold for phash (lower=stricter).
  SLIDE_THUMB_W=64           Thumbnail width for slide comparison.
  SLIDE_THUMB_H=36           Thumbnail height for slide comparison.
  OCR_SNIPPET_CHARS=80       Max OCR chars per frame in LLM prompts.
  PHI3_SINGLE_CALL=true      Merge metadata+notes into one LLM call.
  PHI3_MAX_TOKENS_SINGLE_CALL=1000  Token budget for single-call mode.
  INDEX_WRITE_INTERVAL=50    Frames between lazy index flushes.
  OCR_BATCH_SIZE=16          Frames per OCR batch call.
  TRANSFORMERS_OFFLINE=1     Block all HuggingFace network calls (offline mode).
  HF_DATASETS_OFFLINE=1      Block all HuggingFace dataset downloads.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Frame extraction ──────────────────────────────────────────────────────
    fps:           float = float(os.environ.get("PIPELINE_FPS",    "0.5"))
    min_ocr_chars: int   = int(os.environ.get("MIN_OCR_CHARS",     "20"))

    # ── OCR ───────────────────────────────────────────────────────────────────
    ocr_confidence_threshold: float = float(
        os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.4")
    )

    @property
    def telemetry_confidence_threshold(self) -> float:
        return self.ocr_confidence_threshold

    # ── v3.3.0: OCR backend preference ───────────────────────────────────────
    # Set PADDLEOCR_DISABLE=true to force EasyOCR even when PaddleOCR is
    # installed.  Checked at runtime in _make_ocr_extractor().
    paddleocr_disable: bool = (
        os.environ.get("PADDLEOCR_DISABLE", "false").lower() == "true"
    )

    # v3.3.0: max OCR text chars per frame fed into LLM prompts.
    # Longer snippets waste tokens without improving topic extraction.
    ocr_snippet_chars: int = int(os.environ.get("OCR_SNIPPET_CHARS", "80"))

    # ── v3.3.0: Slide detection ───────────────────────────────────────────────
    # Perceptual hash (phash) threshold — Hamming distance.
    # 0 = identical images, 64 = completely different.
    # 8 is a good default: catches slide transitions, ignores minor flicker.
    phash_threshold: int = int(os.environ.get("PHASH_THRESHOLD", "8"))

    # Thumbnail dimensions for slide comparison (smaller = faster).
    # 64×36 is ~100× fewer pixels than 1280×720 with negligible accuracy loss.
    slide_thumb_w: int = int(os.environ.get("SLIDE_THUMB_W", "64"))
    slide_thumb_h: int = int(os.environ.get("SLIDE_THUMB_H", "36"))

    # Disable imagehash (force histogram fallback)
    imagehash_disable: bool = (
        os.environ.get("IMAGEHASH_DISABLE", "false").lower() == "true"
    )

    # ── LLM / Reasoning ──────────────────────────────────────────────────────
    reasoning_model_id:   str = os.environ.get("REASONING_MODEL_ID",    "models/phi3mini")
    max_reasoning_tokens: int = int(os.environ.get("MAX_REASONING_TOKENS", "1024"))

    # ── Phi-3-mini specific ───────────────────────────────────────────────────
    phi3_adapter_path: str  = os.environ.get("PHI3_ADAPTER_PATH", "")
    phi3_load_in_4bit: bool = (
        os.environ.get("PHI3_LOAD_IN_4BIT", "true").lower() == "true"
    )

    # v3.3.0: merge metadata + notes into one LLM call to save round-trip.
    # Useful when the model is fast (GGUF / llama.cpp) and the 4096-token
    # context is sufficient.  Disabled by default for safety.
    phi3_single_call: bool = (
        os.environ.get("PHI3_SINGLE_CALL", "false").lower() == "true"
    )

    # ── Whisper ───────────────────────────────────────────────────────────────
    # v3.1.0: "tiny" is ~4× faster than "base" with sufficient accuracy.
    whisper_model_size: str = os.environ.get("WHISPER_MODEL_SIZE", "tiny")

    # v3.1.0: hard wall-clock timeout for transcription.
    whisper_timeout_sec: int = int(os.environ.get("WHISPER_TIMEOUT_SEC", "600"))

    # v3.3.0: use all available CPU cores for Whisper CPU inference.
    # Previously hard-coded to 4; auto-detect is almost always better.
    whisper_cpu_threads: int = int(
        os.environ.get("WHISPER_CPU_THREADS", str(os.cpu_count() or 4))
    )

    # ── Storage ───────────────────────────────────────────────────────────────
    output_dir: str = os.environ.get("OUTPUT_DIR", "outputs")

    # v3.3.0: RAM disk path for hot-path frame writes (Linux only).
    # The pipeline auto-detects availability; this is exposed here for
    # completeness and for override via RAMDISK_PATH env var.
    ramdisk_path: str  = os.environ.get("RAMDISK_PATH", "/dev/shm/academic_frames")
    ramdisk_enable: bool = (
        os.environ.get("RAMDISK_ENABLE", "auto").lower() != "false"
    )

    # ── Processing limits ─────────────────────────────────────────────────────
    max_frames_in_memory:  int = int(os.environ.get("MAX_FRAMES_IN_MEMORY",  "2000"))
    notes_sample_frames:   int = int(os.environ.get("NOTES_SAMPLE_FRAMES",   "15"))
    summary_sample_frames: int = int(os.environ.get("SUMMARY_SAMPLE_FRAMES", "10"))

    # v3.1.0: hard cap on total frames extracted per video stream.
    max_frames_extract: int = int(os.environ.get("MAX_FRAMES_EXTRACT", "720"))

    # v3.1.0: max frames fed into LLM prompts during Phase 2.
    phase2_max_frames: int = int(os.environ.get("PHASE2_MAX_FRAMES", "40"))

    # v3.3.0: frame index lazy-write interval.
    # Index is only flushed to disk every N frames (and at stream end).
    # Reducing disk write frequency cuts I/O on spinning/network drives.
    # Was implicit 10 in the old pipeline; default raised to 50.
    index_write_interval: int = int(os.environ.get("INDEX_WRITE_INTERVAL", "50"))

    # ── v3.1.0: Adaptive FPS ─────────────────────────────────────────────────
    fps_long_threshold_1: int   = int(os.environ.get("FPS_LONG_THRESHOLD_1", str(20 * 60)))
    fps_long_threshold_2: int   = int(os.environ.get("FPS_LONG_THRESHOLD_2", str(40 * 60)))
    fps_for_long_1:       float = float(os.environ.get("FPS_FOR_LONG_1", "0.2"))
    fps_for_long_2:       float = float(os.environ.get("FPS_FOR_LONG_2", "0.1"))

    # ── v3.3.0: OCR batch size (raised 8 → 16) ───────────────────────────────
    # 16 frames per batch halves GPU round-trips vs the v3.1.0 default of 8.
    # Memory impact is negligible (each frame is a small numpy array).
    # Override: OCR_BATCH_SIZE=8 to revert if VRAM is tight.
    ocr_batch_size: int = int(os.environ.get("OCR_BATCH_SIZE", "16"))

    # Number of async OCR worker threads in the thread pool.
    ocr_workers: int = int(os.environ.get("OCR_WORKERS", "2"))

    # ── v3: Slide change detection ────────────────────────────────────────────
    # These are used as fallback values when imagehash is not available.
    slide_hist_threshold: float = float(os.environ.get("SLIDE_HIST_THRESHOLD", "0.92"))
    slide_ssim_threshold: float = float(os.environ.get("SLIDE_SSIM_THRESHOLD", "0.85"))
    slide_min_seconds:    float = float(os.environ.get("SLIDE_MIN_SECONDS",    "1.0"))

    # ── v3: Context window ────────────────────────────────────────────────────
    phi3_context_length: int = int(os.environ.get("PHI3_CONTEXT_LENGTH", "4096"))

    # ── Per-task token limits ─────────────────────────────────────────────────
    max_tokens_notes:      int = int(os.environ.get("MAX_TOKENS_NOTES",       "4096"))
    max_tokens_flashcards: int = int(os.environ.get("MAX_TOKENS_FLASHCARDS",  "3072"))
    max_tokens_quiz:       int = int(os.environ.get("MAX_TOKENS_QUIZ",        "2048"))
    max_tokens_summary:    int = int(os.environ.get("MAX_TOKENS_SUMMARY",     "1024"))
    max_tokens_frame:      int = int(os.environ.get("MAX_TOKENS_FRAME",        "400"))
    max_tokens_everything: int = int(os.environ.get("MAX_TOKENS_EVERYTHING",  "1200"))

    # v3.0.3: raised 900 → 1200
    max_tokens_transcript_only: int = int(
        os.environ.get("MAX_TOKENS_TRANSCRIPT_ONLY", "1200")
    )

    # v3.3.0: token budget for merged single-call Phase 2
    max_tokens_single_call: int = int(
        os.environ.get("PHI3_MAX_TOKENS_SINGLE_CALL", "1000")
    )

    # ── Per-call token overrides (used by main.py helpers) ───────────────────
    max_tokens_meta:  int = int(os.environ.get("PHI3_MAX_TOKENS_META",  "300"))
    max_tokens_notes2: int = int(os.environ.get("PHI3_MAX_TOKENS_NOTES", "700"))
    max_tokens_cards: int = int(os.environ.get("PHI3_MAX_TOKENS_CARDS", "1100"))

    # ── OCR noise filter ──────────────────────────────────────────────────────
    # v3.0.3 FIX: lowered from 15 → 8 → 30.
    min_ocr_word_chars: int = int(os.environ.get("MIN_OCR_WORD_CHARS", "30"))

    # ── EasyOCR GPU flag ──────────────────────────────────────────────────────
    # Note: "false" default is intentional — EasyOCR GPU can conflict with
    # Phi-3 4-bit CUDA memory on RTX 3050 (4 GB). PaddleOCR is more
    # memory-efficient and handles GPU sharing better.
    easyocr_gpu: bool = (
        os.environ.get("EASYOCR_GPU", "false").lower() == "true"
    )

    # ── Offline mode ──────────────────────────────────────────────────────────
    # When true, all HuggingFace network calls are blocked. Models must be
    # pre-downloaded. Set via TRANSFORMERS_OFFLINE=1 env var (standard HF
    # convention — config reads it here for documentation purposes only;
    # the HF library reads it directly from the environment).
    @property
    def transformers_offline(self) -> bool:
        return os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"

    @property
    def hf_datasets_offline(self) -> bool:
        return os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"


# Singleton
config = Config()