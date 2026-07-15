"""
academic_system/whisper_transcriber1.py
========================================
Audio extraction + transcription for the Academic Intelligence System.

v3.3.0 — faster-whisper backend with streaming (chunked mode removed)
-----------------------------------------------------------------------
PRIMARY BACKEND: faster-whisper (CTranslate2)
  • 4× faster than openai/whisper on the same hardware
  • INT8 quantisation: ~300 MB VRAM for tiny, ~600 MB for base
  • Streams segments as a generator — never loads full audio into GPU memory
  • VAD filter built-in: skips silence, prevents hallucinations on long pauses

FALLBACK: openai-whisper
  If faster-whisper is not installed, falls back to openai/whisper with
  explicit WAV chunking (_CHUNK_DURATION_SEC pieces) and per-chunk
  timeouts so a 1-hour file is never processed in one blocking call.

Install:
  pip install faster-whisper        # primary (recommended)
  pip install soundfile              # needed for fallback WAV chunking
  pip install openai-whisper         # fallback only

VRAM budget on RTX 3050 (4 GB):
  Phi-3 4-bit          ~2.4 GB
  EasyOCR GPU          ~0.3 GB
  faster-whisper tiny  ~0.3 GB  (INT8)
  OS + driver          ~0.3 GB
  ─────────────────────────────
  Total                ~3.3 GB  ✓ fits with ~700 MB headroom

Public API (unchanged):
  video_has_audio(video_path)   → bool
  extract_audio(video_path, out)→ Optional[str]
  transcribe(audio_path, ...)   → Dict
  convert_to_wav(src, dst)      → str
"""
from __future__ import annotations

import logging
import os
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── faster-whisper (primary) ──────────────────────────────────────────────────
try:
    from faster_whisper import WhisperModel as _FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
    logger.info("[Whisper] faster-whisper available ✓  (primary backend)")
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.warning(
        "[Whisper] faster-whisper not installed — will use openai-whisper fallback.\n"
        "  For best performance on RTX 3050: pip install faster-whisper"
    )

# ── openai-whisper (fallback) ─────────────────────────────────────────────────
try:
    import whisper as _openai_whisper_lib
    OPENAI_WHISPER_AVAILABLE = True
    logger.info("[Whisper] openai-whisper available ✓  (fallback backend)")
except ImportError:
    OPENAI_WHISPER_AVAILABLE = False
    logger.warning("[Whisper] openai-whisper not installed.")

# Public flag — True if either backend is available
WHISPER_AVAILABLE = FASTER_WHISPER_AVAILABLE or OPENAI_WHISPER_AVAILABLE

# ── soundfile for WAV chunking (fallback path only) ───────────────────────────
try:
    import soundfile as _sf
    import numpy as _np
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

# ── Configuration (all overridable via env vars) ───────────────────────────────
_MODEL_SIZE:        str = os.environ.get("WHISPER_MODEL_SIZE",          "tiny")
_CHUNK_TIMEOUT_SEC: int = int(os.environ.get("WHISPER_CHUNK_TIMEOUT_SEC", "300"))
_TOTAL_TIMEOUT_SEC: int = int(os.environ.get("WHISPER_TIMEOUT_SEC",      "600"))
_CHUNK_DURATION_SEC:int = int(os.environ.get("WHISPER_CHUNK_DURATION_SEC","300"))
_GPU_COMPUTE_TYPE:  str = os.environ.get("FASTER_WHISPER_COMPUTE_TYPE", "int8_float16")
_CPU_COMPUTE_TYPE:  str = "int8"

# Model caches
_fw_model_cache: Dict[str, Any] = {}
_ow_model_cache: Dict[str, Any] = {}


def _progress_bar(current: float, total: float, width: int = 28) -> str:
    """Render a simple ASCII progress bar for terminal logs."""
    if total <= 0:
        return "[" + ("-" * width) + "]"
    ratio = max(0.0, min(1.0, current / total))
    filled = int(ratio * width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _log_progress(prefix: str, current: float, total: float, unit: str = "", extra: str = "") -> None:
    """Log a compact terminal-friendly progress line."""
    bar = _progress_bar(current, total)
    pct = 0.0 if total <= 0 else max(0.0, min(100.0, (current / total) * 100.0))
    if total > 0:
        msg = f"{prefix} {bar} {pct:5.1f}% ({current:.1f}/{total:.1f}{unit})"
    else:
        msg = f"{prefix} {bar} {current:.1f}{unit}"
    if extra:
        msg = f"{msg} | {extra}"
    logger.info(msg)


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
    logger.info(f"[Whisper] Extracting audio from {Path(video_path).name}...")

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
    logger.info(f"[Whisper] Converting {Path(src_path).name} to 16 kHz mono WAV...")
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


def _get_audio_duration(audio_path: str) -> float:
    """Get duration of an audio file in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", audio_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except Exception as exc:
        logger.warning(f"[Whisper] Could not determine duration: {exc}")
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  FASTER-WHISPER BACKEND  (primary)
# ──────────────────────────────────────────────────────────────────────────────

def _load_faster_whisper(model_size: str, device: str) -> Any:
    """
    Load (or return cached) a faster-whisper WhisperModel.

    VRAM usage with INT8 quantisation:
      tiny  INT8 → ~300 MB   base  INT8 → ~500 MB   small INT8 → ~900 MB
    """
    cache_key = f"{model_size}:{device}"
    if cache_key in _fw_model_cache:
        return _fw_model_cache[cache_key]

    compute_type = _GPU_COMPUTE_TYPE if device == "cuda" else _CPU_COMPUTE_TYPE
    logger.info(
        f"[Whisper] Loading faster-whisper '{model_size}' "
        f"device={device} compute_type={compute_type} …"
    )
    model = _FasterWhisperModel(
        model_size,
        device       = device,
        compute_type = compute_type,
        cpu_threads  = int(os.environ.get("WHISPER_CPU_THREADS", "4")),
    )
    _fw_model_cache[cache_key] = model
    logger.info(f"[Whisper] faster-whisper '{model_size}' loaded ✓")
    return model


def _transcribe_faster_whisper(
    audio_path: str,
    model_size: str,
    language:   Optional[str],
    device:     str,
) -> Dict[str, Any]:
    """
    Transcribe using faster-whisper with segment-level streaming + timeouts.

    The generator runs in a daemon thread. The main thread collects
    segments with a per-segment timeout and a hard wall-clock limit.

    Key settings:
      beam_size=1              greedy decoding — 3× faster than beam_size=5
      vad_filter=True          skip silence, stop hallucination on pauses
      condition_on_previous_text=False   prevent repetition loops on long audio
      word_timestamps=False    not needed for study notes; saves ~20% time
    """
    model = _load_faster_whisper(model_size, device)
    total_duration = _get_audio_duration(audio_path)

    segments_out:   List[Dict[str, Any]] = []
    detected_lang:  str = language or "en"
    segment_queue:  List[Any] = []
    error_holder:   List[Optional[Exception]] = [None]

    def _generator_thread():
        try:
            segments_gen, info = model.transcribe(
                audio_path,
                language                   = language,
                beam_size                  = 1,
                vad_filter                 = True,
                vad_parameters             = {
                    "min_silence_duration_ms": 500,
                    "threshold":               0.5,
                },
                condition_on_previous_text = False,
                word_timestamps            = False,
            )
            segment_queue.append(("lang", info.language))
            for seg in segments_gen:
                segment_queue.append(("seg", seg))
            segment_queue.append(("done", None))
        except Exception as exc:
            segment_queue.append(("err", exc))

    t = threading.Thread(target=_generator_thread, daemon=True, name="fw_transcribe")
    t.start()

    wall_start     = time.monotonic()
    last_seg_time  = time.monotonic()
    full_text_parts: List[str] = []

    while True:
        # ── Hard overall timeout ──────────────────────────────────────────────
        if time.monotonic() - wall_start > _TOTAL_TIMEOUT_SEC:
            logger.error(
                f"[Whisper] Total timeout ({_TOTAL_TIMEOUT_SEC}s) — "
                "returning partial transcript."
            )
            break

        # ── Per-segment (chunk) timeout ───────────────────────────────────────
        if time.monotonic() - last_seg_time > _CHUNK_TIMEOUT_SEC:
            logger.error(
                f"[Whisper] No segment for {_CHUNK_TIMEOUT_SEC}s — "
                "model appears stuck. Returning partial transcript."
            )
            break

        if segment_queue:
            kind, value = segment_queue.pop(0)

            if kind == "lang":
                detected_lang = value
                logger.info(f"[Whisper] Language detected: '{detected_lang}'")

            elif kind == "seg":
                seg  = value
                text = seg.text.strip()
                if text:
                    segments_out.append({
                        "id":         len(segments_out),
                        "start":      round(float(seg.start), 3),
                        "end":        round(float(seg.end),   3),
                        "text":       text,
                        "confidence": round(
                            1.0 - float(getattr(seg, "no_speech_prob", 0.0)), 4
                        ),
                    })
                    full_text_parts.append(text)
                    last_seg_time = time.monotonic()
                    if total_duration > 0:
                        _log_progress(
                            "[Whisper] Streaming",
                            min(float(seg.end), total_duration),
                            total_duration,
                            unit="s",
                            extra=f"segments={len(segments_out)}",
                        )
                    logger.debug(
                        f"[Whisper] [{seg.start:.1f}s→{seg.end:.1f}s] {text[:60]}"
                    )

            elif kind == "done":
                logger.info(
                    f"[Whisper] faster-whisper complete: "
                    f"{len(segments_out)} segments in "
                    f"{time.monotonic()-wall_start:.1f}s"
                )
                break

            elif kind == "err":
                error_holder[0] = value
                logger.error(f"[Whisper] Generator error: {value}")
                break
        else:
            time.sleep(0.05)  # avoid busy-wait

    result: Dict[str, Any] = {
        "text":              " ".join(full_text_parts).strip(),
        "language":          detected_lang,
        "segments":          segments_out,
        "whisper_available": True,
        "backend":           "faster-whisper",
        "model":             model_size,
    }
    if error_holder[0] is not None:
        result["error"] = str(error_holder[0])

    logger.info(
        f"[Whisper] Done: {len(segments_out)} segs, "
        f"{len(result['text'])} chars, lang='{detected_lang}', "
        f"backend=faster-whisper/{model_size}/"
        f"{'INT8' if 'int8' in _GPU_COMPUTE_TYPE else 'FP16'}"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
#  OPENAI-WHISPER FALLBACK  (chunked, per-chunk timeout)
# ──────────────────────────────────────────────────────────────────────────────

def _split_wav_chunks(audio_path: str, chunk_sec: int) -> List[str]:
    """
    Split a WAV into fixed-length chunks using soundfile.
    Returns list of temp WAV paths. Falls back to [audio_path] if soundfile
    is not available.
    """
    if not SOUNDFILE_AVAILABLE:
        logger.warning(
            "[Whisper] soundfile not installed — processing full file. "
            "This may be slow for long lectures. pip install soundfile"
        )
        return [audio_path]

    try:
        data, sr = _sf.read(audio_path, dtype="float32")
    except Exception as exc:
        logger.warning(f"[Whisper] soundfile read failed ({exc}) — full file.")
        return [audio_path]

    chunk_samples = int(chunk_sec * sr)
    chunk_paths:  List[str] = []
    stem   = Path(audio_path).stem
    parent = Path(audio_path).parent

    for i, start in enumerate(range(0, len(data), chunk_samples)):
        chunk = data[start : start + chunk_samples]
        if len(chunk) == 0:
            continue
        cp = str(parent / f"{stem}_chunk{i:04d}.wav")
        _sf.write(cp, chunk, sr, subtype="PCM_16")
        chunk_paths.append(cp)

    logger.info(
        f"[Whisper] Split into {len(chunk_paths)} chunk(s) of {chunk_sec}s."
    )
    return chunk_paths


def _transcribe_chunk_with_timeout(
    model: Any, chunk_path: str, language: Optional[str], timeout_sec: int,
) -> Dict[str, Any]:
    """Transcribe one WAV chunk with a hard per-chunk timeout."""
    result_holder: List[Any]             = [None]
    exc_holder:    List[Optional[Exception]] = [None]

    def _target():
        try:
            result_holder[0] = model.transcribe(
                chunk_path, language=language, fp16=False, verbose=False,
            )
        except Exception as exc:
            exc_holder[0] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        logger.warning(f"[Whisper] Chunk timeout ({timeout_sec}s) for {chunk_path}.")
        return {}
    if exc_holder[0]:
        logger.error(f"[Whisper] Chunk error: {exc_holder[0]}")
        return {}
    return result_holder[0] or {}


def _transcribe_openai_whisper_chunked(
    audio_path: str,
    model_size: str,
    language:   Optional[str],
) -> Dict[str, Any]:
    """openai-whisper fallback: chunk → transcribe each → stitch."""
    global _ow_model_cache
    if model_size not in _ow_model_cache:
        logger.info(f"[Whisper] Loading openai-whisper '{model_size}' …")
        _ow_model_cache[model_size] = _openai_whisper_lib.load_model(model_size)
        logger.info(f"[Whisper] openai-whisper '{model_size}' loaded ✓")

    model       = _ow_model_cache[model_size]
    chunk_paths = _split_wav_chunks(audio_path, _CHUNK_DURATION_SEC)
    is_chunked  = len(chunk_paths) > 1

    all_segments:  List[Dict]  = []
    all_text:      List[str]   = []
    detected_lang: str         = language or "en"
    seg_id_offset: int         = 0
    wall_start = time.monotonic()

    for idx, chunk_path in enumerate(chunk_paths):
        if time.monotonic() - wall_start > _TOTAL_TIMEOUT_SEC:
            logger.error(
                f"[Whisper] Overall timeout — stopped after {idx}/{len(chunk_paths)} chunks."
            )
            break

        logger.info(f"[Whisper] Chunk {idx+1}/{len(chunk_paths)}: {chunk_path}")
        raw = _transcribe_chunk_with_timeout(
            model, chunk_path, language, _CHUNK_TIMEOUT_SEC
        )
        if not raw:
            continue

        if idx == 0:
            detected_lang = raw.get("language", language or "en")

        text = raw.get("text", "").strip()
        if text:
            all_text.append(text)

        offset = idx * _CHUNK_DURATION_SEC
        for seg in raw.get("segments", []):
            all_segments.append({
                "id":         seg_id_offset + seg["id"],
                "start":      round(float(seg["start"]) + offset, 3),
                "end":        round(float(seg["end"])   + offset, 3),
                "text":       seg["text"].strip(),
                "confidence": round(1.0 - float(seg.get("no_speech_prob", 0.0)), 4),
            })
        seg_id_offset += len(raw.get("segments", []))

    # Clean up temp chunks
    if is_chunked:
        for cp in chunk_paths:
            try:
                if cp != audio_path:
                    os.remove(cp)
            except OSError:
                pass

    full_text = " ".join(all_text).strip()
    logger.info(
        f"[Whisper] openai-whisper done: {len(all_segments)} segs, "
        f"{len(chunk_paths)} chunk(s), lang='{detected_lang}'"
    )
    return {
        "text":              full_text,
        "language":          detected_lang,
        "segments":          all_segments,
        "whisper_available": True,
        "backend":           "openai-whisper",
        "model":             model_size,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def transcribe(
    audio_path:  str,
    language:    Optional[str] = None,
    model_size:  str           = _MODEL_SIZE,
) -> Dict[str, Any]:
    """
    Transcribe a WAV audio file using the best available backend.

    Backend priority:
      1. faster-whisper  — INT8 GPU, segment streaming, VAD filter
      2. openai-whisper  — chunked into _CHUNK_DURATION_SEC pieces

    Parameters
    ----------
    audio_path : str       16 kHz mono WAV path.
    language   : str|None  ISO 639-1 code or None for auto-detect.
    model_size : str       "tiny"|"base"|"small"|"medium"|"large"
                           Default: "tiny" (env WHISPER_MODEL_SIZE).

    Returns
    -------
    dict with keys: text, language, segments, whisper_available,
                    backend, model, error (only on failure)
    """
    if not WHISPER_AVAILABLE:
        return {
            "text": "", "language": language or "en", "segments": [],
            "whisper_available": False, "backend": "none",
            "error": "Neither faster-whisper nor openai-whisper is installed.\n"
                     "  Recommended: pip install faster-whisper",
        }

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    logger.info(
        f"[Whisper] Transcribing {Path(audio_path).name} | "
        f"model={model_size} lang={language or 'auto'} device={device} | "
        f"backend={'faster-whisper' if FASTER_WHISPER_AVAILABLE else 'openai-whisper'}"
    )

    if FASTER_WHISPER_AVAILABLE:
        try:
            return _transcribe_faster_whisper(audio_path, model_size, language, device)
        except Exception as exc:
            logger.error(f"[Whisper] faster-whisper failed: {exc}", exc_info=True)
            if not OPENAI_WHISPER_AVAILABLE:
                return {
                    "text": "", "language": language or "en", "segments": [],
                    "whisper_available": True, "backend": "faster-whisper",
                    "error": str(exc),
                }

    try:
        return _transcribe_openai_whisper_chunked(audio_path, model_size, language)
    except Exception as exc:
        logger.error(f"[Whisper] openai-whisper fallback failed: {exc}", exc_info=True)
        return {
            "text": "", "language": language or "en", "segments": [],
            "whisper_available": True, "backend": "openai-whisper",
            "error": str(exc),
        }


# ──────────────────────────────────────────────────────────────────────────────
#  VRAM RELEASE  — free GPU memory after transcription is done
# ──────────────────────────────────────────────────────────────────────────────

def release_models() -> None:
    """
    Explicitly delete all cached Whisper models and free GPU memory.

    On RTX 3050 (4 GB VRAM) the pipeline sequence is:
      1. Whisper transcription  (~300 MB VRAM for tiny INT8)
      2. Phi-3 model loading    (~2.5 GB VRAM for 4-bit)

    If the Whisper model stays resident, Phi-3 loading hits peak VRAM
    allocation (~3.0–3.5 GB during quantisation) and Windows silently
    terminates the process (OOM kill — no Python exception, no log).

    Call this AFTER all Whisper futures have been collected and BEFORE
    Phi-3 loads.
    """
    global _fw_model_cache, _ow_model_cache

    released = []

    # ── faster-whisper (CTranslate2) ──────────────────────────────────────────
    if _fw_model_cache:
        for key in list(_fw_model_cache.keys()):
            try:
                model = _fw_model_cache[key]
                # Force CTranslate2 to release its internal GPU allocator
                if hasattr(model, 'unload_model'):
                    model.unload_model()
                del _fw_model_cache[key]
                released.append(f"faster-whisper:{key}")
            except Exception as e:
                logger.warning(f"[Whisper/Debug] Error deleting faster-whisper model {key}: {e}")
        _fw_model_cache.clear()

    # ── openai-whisper (PyTorch) ──────────────────────────────────────────────
    if _ow_model_cache:
        for key in list(_ow_model_cache.keys()):
            try:
                del _ow_model_cache[key]
                released.append(f"openai-whisper:{key}")
            except Exception:
                pass
        _ow_model_cache.clear()

    # ── Force Python GC + CUDA cache flush ────────────────────────────────────
    import gc
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            vfree, vtotal = torch.cuda.mem_get_info()
            logger.info(
                f"[Whisper] Models released: {released}. "
                f"VRAM now: {vfree/(1024**3):.1f}GB free of {vtotal/(1024**3):.1f}GB total."
            )
        else:
            logger.info(f"[Whisper] Models released: {released} (CPU mode).")
    except ImportError:
        logger.info(f"[Whisper] Models released: {released}.")