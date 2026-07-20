"""
academic_system/whisper_transcriber_api.py
===========================================
Drop-in replacement for the local Whisper transcription in
whisper_transcriber1.py that calls a Whisper model through Hugging Face's
Inference API instead of loading faster-whisper/openai-whisper locally.

Use this when deploying to a host with limited RAM/no GPU (e.g. Render's
free/starter CPU instances), where loading Whisper alongside EasyOCR and
SentenceTransformer in the same process risks an out-of-memory crash.

Same public interface as whisper_transcriber1.transcribe():
    transcribe(audio_path, language=None, model_size=...) -> dict with
    keys: text, language, segments, whisper_available, backend, model

Requires:
    pip install huggingface_hub
    env var HF_TOKEN set to a Hugging Face access token

Optional env vars:
    WHISPER_HF_MODEL_ID   — override the model repo id (default below)

Notes / limitations vs. the local backend:
    - No word/segment-level timestamps are returned by the standard HF ASR
      task for most models — 'segments' comes back as a single segment
      spanning the whole clip. If your pipeline needs per-segment timing
      elsewhere, that's the one behavioural difference to be aware of.
    - HF's free serverless Inference API enforces a payload size / duration
      limit (varies by model/plan). Very long audio may need to be chunked
      client-side (split into ~60–120s WAV pieces and concatenated) if you
      hit a 413/422 error — this module does not chunk automatically.
"""
from __future__ import annotations

import logging
import os
import wave
import contextlib
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning(
        "[Whisper/API] huggingface_hub not installed. Run: pip install huggingface_hub"
    )

_HF_MODEL_ID = os.environ.get("WHISPER_HF_MODEL_ID", "openai/whisper-large-v3-turbo")

WHISPER_AVAILABLE = HF_HUB_AVAILABLE
FASTER_WHISPER_AVAILABLE = False   # kept for import compatibility with main_v7-9.py
OPENAI_WHISPER_AVAILABLE = False   # kept for import compatibility with main_v7-9.py


def _get_audio_duration(audio_path: str) -> float:
    try:
        with contextlib.closing(wave.open(audio_path, "rb")) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate) if rate else 0.0
    except Exception:
        return 0.0


def _get_client() -> "InferenceClient":
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "[Whisper/API] HF_TOKEN environment variable not set. "
            "Create a token at https://huggingface.co/settings/tokens "
            "and set it as a secret in your deployment environment."
        )
    return InferenceClient(model=_HF_MODEL_ID, token=hf_token)


def transcribe(
    audio_path: str,
    language:   Optional[str] = None,
    model_size: str = "",   # accepted for signature compat, unused — API decides the model
) -> Dict[str, Any]:
    """
    Transcribe a WAV audio file via the Hugging Face Inference API.

    Same return contract as whisper_transcriber1.transcribe():
        text, language, segments, whisper_available, backend, model
    """
    if not HF_HUB_AVAILABLE:
        return {
            "text": "", "language": language or "en", "segments": [],
            "whisper_available": False, "backend": "none",
            "error": "huggingface_hub is not installed.\n  Run: pip install huggingface_hub",
        }

    duration = _get_audio_duration(audio_path)
    logger.info(
        f"[Whisper/API] Transcribing via HF Inference API | model={_HF_MODEL_ID} "
        f"lang={language or 'auto'} duration={duration:.1f}s"
    )

    try:
        client = _get_client()
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        output = client.automatic_speech_recognition(audio_bytes, model=_HF_MODEL_ID)

        # huggingface_hub returns either a plain string, or an object/dict
        # with a `.text` / ["text"] field depending on version — handle both.
        if isinstance(output, str):
            text = output.strip()
        elif isinstance(output, dict):
            text = str(output.get("text", "")).strip()
        else:
            text = str(getattr(output, "text", "")).strip()

        result: Dict[str, Any] = {
            "text": text,
            "language": language or "auto",
            # Most HF ASR endpoints don't return per-segment timestamps —
            # give downstream code one segment spanning the full clip so
            # anything expecting a non-empty `segments` list still works.
            "segments": [{
                "id": 0,
                "start": 0.0,
                "end": round(duration, 3),
                "text": text,
                "confidence": 1.0,
            }] if text else [],
            "whisper_available": True,
            "backend": "hf-inference-api",
            "model": _HF_MODEL_ID,
        }

        logger.info(
            f"[Whisper/API] Done: {len(text)} chars via {_HF_MODEL_ID}"
        )
        return result

    except Exception as exc:
        logger.error(f"[Whisper/API] Transcription failed: {exc}", exc_info=True)
        return {
            "text": "", "language": language or "en", "segments": [],
            "whisper_available": True, "backend": "hf-inference-api",
            "model": _HF_MODEL_ID, "error": str(exc),
        }


def release_models() -> None:
    """No-op — nothing is held in local memory with the API backend."""
    pass