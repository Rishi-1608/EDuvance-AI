"""
academic_system/whisper_transcriber.py
========================================
Handles audio extraction from video and Whisper-based transcription.

v3 change: language parameter now accepts None for auto-detection.
When language=None, Whisper detects the language automatically and
returns it in the result dict — consumed by LanguageDetector.from_whisper().

Functions:
  video_has_audio(video_path)       → bool
  extract_audio(video_path, out)    → Optional[str]   WAV path or None
  transcribe(audio_path, ...)       → Dict             structured transcript
  convert_to_wav(src, dst)          → str              converted WAV path
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import whisper as _whisper_lib
    WHISPER_AVAILABLE = True
    logger.info("[Whisper] openai-whisper available ✓")
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning(
        "[Whisper] openai-whisper not installed — transcription disabled. "
        "Install with: pip install openai-whisper"
    )

_whisper_model_cache: Dict[str, Any] = {}


# ──────────────────────────────────────────────────────────────────────────────
#  AUDIO EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def video_has_audio(video_path: str) -> bool:
    """Return True if the video file contains at least one audio stream."""
    try:
        result = subprocess.run(
            ["ffprobe", "-i", str(video_path),
             "-show_streams", "-select_streams", "a", "-loglevel", "error"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        return len(result.stdout.strip()) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"[Whisper] ffprobe check failed: {exc}")
        return False


def extract_audio(video_path: str, audio_dir: str) -> Optional[str]:
    """
    Extract audio from a video as a 16 kHz mono WAV file.
    Returns the WAV path, or None if extraction fails / no audio stream.
    """
    if not video_has_audio(video_path):
        logger.info(f"[Whisper] No audio stream in {video_path}.")
        return None

    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, f"{Path(video_path).stem}.wav")

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
             audio_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, timeout=600,
        )
        logger.info(f"[Whisper] Audio extracted → {audio_path}")
        return audio_path
    except subprocess.CalledProcessError as exc:
        logger.error(f"[Whisper] ffmpeg failed: {exc.stderr.decode('utf-8', errors='replace')}")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"[Whisper] ffmpeg timed out for {video_path}")
        return None


def convert_to_wav(src_path: str, audio_dir: str) -> str:
    """Convert any audio format to 16 kHz mono WAV. Returns original path on failure."""
    dst_path = os.path.join(audio_dir, f"{Path(src_path).stem}_converted.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src_path),
             "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", dst_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, timeout=300,
        )
        logger.info(f"[Whisper] Converted → {dst_path}")
        return dst_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"[Whisper] Conversion failed ({exc}), using original.")
        return src_path


# ──────────────────────────────────────────────────────────────────────────────
#  TRANSCRIPTION
# ──────────────────────────────────────────────────────────────────────────────

def transcribe(
    audio_path:  str,
    language:    Optional[str] = None,   # None = auto-detect (v3 default)
    model_size:  str = "base",
) -> Dict[str, Any]:
    """
    Transcribe an audio file using Whisper.

    Parameters
    ----------
    audio_path : str
        Path to a 16 kHz mono WAV file.
    language : str | None
        ISO 639-1 code ("en", "hi", …) or None to let Whisper auto-detect.
        Auto-detection is the v3 default — the detected code is returned
        in the result dict and consumed by LanguageDetector.from_whisper().
    model_size : str
        Whisper model: tiny | base | small | medium | large.

    Returns
    -------
    {
      "text":              "Full transcript",
      "language":          "en",          ← always present (detected or passed)
      "segments":          [...],
      "whisper_available": True,
    }
    """
    if not WHISPER_AVAILABLE:
        return {
            "text": "", "language": language or "en",
            "segments": [], "whisper_available": False,
            "error": "openai-whisper not installed",
        }

    global _whisper_model_cache
    if model_size not in _whisper_model_cache:
        logger.info(f"[Whisper] Loading model '{model_size}' …")
        _whisper_model_cache[model_size] = _whisper_lib.load_model(model_size)
        logger.info(f"[Whisper] Model '{model_size}' loaded ✓")

    model = _whisper_model_cache[model_size]

    try:
        logger.info(f"[Whisper] Transcribing {audio_path} (language={language or 'auto'}) …")

        # Pass language=None for auto-detection; Whisper will detect and return it
        raw = model.transcribe(
            audio_path,
            language = language,   # None → auto-detect
            fp16     = False,
            verbose  = False,
        )

        detected_lang = raw.get("language", language or "en")

        segments: List[Dict[str, Any]] = [
            {
                "id":         seg["id"],
                "start":      round(float(seg["start"]), 3),
                "end":        round(float(seg["end"]),   3),
                "text":       seg["text"].strip(),
                "confidence": round(1.0 - float(seg.get("no_speech_prob", 0.0)), 4),
            }
            for seg in raw.get("segments", [])
        ]

        logger.info(
            f"[Whisper] Done — {len(segments)} segment(s), "
            f"detected language='{detected_lang}'"
        )

        return {
            "text":              raw.get("text", "").strip(),
            "language":          detected_lang,
            "segments":          segments,
            "whisper_available": True,
        }

    except Exception as exc:
        logger.error(f"[Whisper] Transcription failed: {exc}")
        return {
            "text": "", "language": language or "en",
            "segments": [], "whisper_available": True,
            "error": str(exc),
        }