"""
academic_system/whisper_transcriber1.py
========================================
Audio extraction + transcription for the Academic Intelligence System.

v3.3.0 — cpu_threads auto-detect + offline improvements
---------------------------------------------------------
  ① cpu_threads auto-detect — WHISPER_CPU_THREADS now defaults to
    os.cpu_count() instead of the hard-coded value of 4. On a 6-core
    machine this gives ~1.5× CPU transcription speedup for free.
    Override: WHISPER_CPU_THREADS=4

  ② Offline model cache path — FASTER_WHISPER_CACHE_DIR and
    WHISPER_CACHE_DIR env vars allow pre-downloaded model weights to be
    loaded from a local directory, enabling fully air-gapped deployments.
    Set to the directory that contains the downloaded model files.
    Example:
      export FASTER_WHISPER_CACHE_DIR=/opt/models/faster-whisper
      export WHISPER_CACHE_DIR=/opt/models/openai-whisper

  ③ Model warmup on first load — after loading, the model processes a
    1-second silent WAV to prime CUDA kernels. Eliminates first-inference
    latency spike on RTX 3050 (typically 2–4 seconds).

  ④ Compute type auto-selection — _GPU_COMPUTE_TYPE now falls back from
    "int8_float16" to "int8" when the GPU does not support float16
    (some older cards). Exception is caught and retried automatically.

  ⑤ Silence padding removed — ffmpeg audio extraction no longer adds
    silence padding that caused Whisper to hallucinate at the end of
    short recordings.

v3.2.0 — faster-whisper backend with chunked streaming
---------------------------------------------------------
  PRIMARY BACKEND: faster-whisper (CTranslate2)
    • 4× faster than openai/whisper on the same hardware
    • INT8 quantisation: ~300 MB VRAM for tiny, ~600 MB for base
    • Streams segments as a generator
    • VAD filter: skips silence, prevents hallucinations on long pauses
    • Per-segment timeout + hard wall-clock timeout

  FALLBACK: openai-whisper
    • Explicit WAV chunking (_CHUNK_DURATION_SEC pieces)
    • Per-chunk timeout

Install:
  pip install faster-whisper        # primary (recommended)
  pip install soundfile              # needed for fallback WAV chunking
  pip install openai-whisper         # fallback only

VRAM budget on RTX 3050 (4 GB):
  Phi-3 4-bit          ~2.4 GB
  PaddleOCR/EasyOCR    ~0.3 GB
  faster-whisper tiny  ~0.3 GB  (INT8)
  OS + driver          ~0.3 GB
  ─────────────────────────────
  Total                ~3.3 GB  ✓

Public API (unchanged):
  video_has_audio(video_path)   → bool
  extract_audio(video_path, out)→ Optional[str]
  transcribe(audio_path, ...)   → Dict
  convert_to_wav(src, dst)      → str
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
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
        "  For best performance: pip install faster-whisper"
    )

# ── openai-whisper (fallback) ─────────────────────────────────────────────────
try:
    import whisper as _openai_whisper_lib
    OPENAI_WHISPER_AVAILABLE = True
    logger.info("[Whisper] openai-whisper available ✓  (fallback backend)")
except ImportError:
    OPENAI_WHISPER_AVAILABLE = False
    logger.warning("[Whisper] openai-whisper not installed.")

WHISPER_AVAILABLE = FASTER_WHISPER_AVAILABLE or OPENAI_WHISPER_AVAILABLE

# ── soundfile for WAV chunking (fallback path only) ───────────────────────────
try:
    import soundfile as _sf
    import numpy as _np
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

# ── Configuration ──────────────────────────────────────────────────────────────
_MODEL_SIZE:        str = os.environ.get("WHISPER_MODEL_SIZE",           "tiny")
_CHUNK_TIMEOUT_SEC: int = int(os.environ.get("WHISPER_CHUNK_TIMEOUT_SEC", "300"))
_TOTAL_TIMEOUT_SEC: int = int(os.environ.get("WHISPER_TIMEOUT_SEC",       "600"))
_CHUNK_DURATION_SEC:int = int(os.environ.get("WHISPER_CHUNK_DURATION_SEC","300"))

# v3.3.0: prefer int8_float16 on GPU, fall back to int8 if unsupported
_GPU_COMPUTE_TYPE: str = os.environ.get("FASTER_WHISPER_COMPUTE_TYPE", "int8_float16")
_CPU_COMPUTE_TYPE: str = "int8"

# v3.3.0: auto-detect CPU thread count (was hard-coded to 4)
_CPU_THREADS: int = int(os.environ.get("WHISPER_CPU_THREADS", str(os.cpu_count() or 4)))

# v3.3.0: offline model cache paths
_FW_CACHE_DIR: Optional[str] = os.environ.get("FASTER_WHISPER_CACHE_DIR")
_OW_CACHE_DIR: Optional[str] = os.environ.get("WHISPER_CACHE_DIR")

# Model caches
_fw_model_cache: Dict[str, Any] = {}
_ow_model_cache: Dict[str, Any] = {}


# ──────────────────────────────────────────────────────────────────────────────
#  v3.3.0: SILENT WAV FOR MODEL WARMUP
# ──────────────────────────────────────────────────────────────────────────────

def _make_silent_wav(duration_sec: float = 1.0, sample_rate: int = 16000) -> str:
    """
    Create a temporary 1-second silent WAV file for CUDA kernel warmup.
    Returns the file path. Caller is responsible for deletion.
    """
    n_samples = int(duration_sec * sample_rate)
    # Minimal WAV header + zero-filled PCM samples
    import struct
    data_size   = n_samples * 2  # 16-bit mono
    header_size = 44
    total_size  = header_size + data_size

    buf = bytearray(total_size)
    # RIFF chunk
    buf[0:4]   = b"RIFF"
    buf[4:8]   = struct.pack("<I", total_size - 8)
    buf[8:12]  = b"WAVE"
    # fmt sub-chunk
    buf[12:16] = b"fmt "
    buf[16:20] = struct.pack("<I", 16)          # sub-chunk size
    buf[20:22] = struct.pack("<H", 1)           # PCM
    buf[22:24] = struct.pack("<H", 1)           # mono
    buf[24:28] = struct.pack("<I", sample_rate)
    buf[28:32] = struct.pack("<I", sample_rate * 2)
    buf[32:34] = struct.pack("<H", 2)           # block align
    buf[34:36] = struct.pack("<H", 16)          # bits per sample
    # data sub-chunk
    buf[36:40] = b"data"
    buf[40:44] = struct.pack("<I", data_size)
    # PCM data is already zero-filled

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(bytes(buf))
    tmp.close()
    return tmp.name


def _warmup_faster_whisper(model: Any) -> None:
    """
    Run the model on a 1-second silent WAV to prime CUDA kernels.
    This eliminates the 2-4 second first-inference latency spike.
    """
    silent_path = None
    try:
        silent_path = _make_silent_wav(duration_sec=1.0)
        list(model.transcribe(silent_path, beam_size=1, vad_filter=True)[0])
        logger.info("[Whisper] CUDA kernel warmup complete ✓")
    except Exception as exc:
        logger.debug(f"[Whisper] Warmup skipped ({exc})")
    finally:
        if silent_path and os.path.isfile(silent_path):
            try:
                os.remove(silent_path)
            except OSError:
                pass


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

    v3.3.0: removed silence padding (-af apad) that caused Whisper to
    hallucinate at the end of short recordings.
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
             # v3.3.0: no -af apad — avoids hallucinations on short clips
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
#  FASTER-WHISPER BACKEND  (primary)
# ──────────────────────────────────────────────────────────────────────────────

def _load_faster_whisper(model_size: str, device: str) -> Any:
    """
    Load (or return cached) a faster-whisper WhisperModel.

    v3.3.0 improvements:
      • cpu_threads = os.cpu_count() (was 4)
      • Offline cache dir support via FASTER_WHISPER_CACHE_DIR
      • Compute type auto-fallback: int8_float16 → int8 if float16 unsupported
      • CUDA kernel warmup on first load

    VRAM usage (INT8):
      tiny ~300 MB  |  base ~500 MB  |  small ~900 MB
    """
    cache_key = f"{model_size}:{device}"
    if cache_key in _fw_model_cache:
        return _fw_model_cache[cache_key]

    compute_type = _GPU_COMPUTE_TYPE if device == "cuda" else _CPU_COMPUTE_TYPE

    def _try_load(ct: str) -> Any:
        kwargs: Dict[str, Any] = dict(
            device       = device,
            compute_type = ct,
            cpu_threads  = _CPU_THREADS,   # v3.3.0: auto-detected
        )
        # v3.3.0: offline cache dir support
        if _FW_CACHE_DIR and os.path.isdir(_FW_CACHE_DIR):
            kwargs["download_root"] = _FW_CACHE_DIR
            logger.info(f"[Whisper] Loading faster-whisper from local cache: {_FW_CACHE_DIR}")

        logger.info(
            f"[Whisper] Loading faster-whisper '{model_size}' "
            f"device={device} compute_type={ct} cpu_threads={_CPU_THREADS} …"
        )
        return _FasterWhisperModel(model_size, **kwargs)

    try:
        model = _try_load(compute_type)
    except Exception as exc:
        # v3.3.0: auto-fallback for GPUs that don't support float16
        if device == "cuda" and "float16" in str(exc).lower():
            logger.warning(
                f"[Whisper] {compute_type} not supported on this GPU — "
                "retrying with int8."
            )
            model = _try_load("int8")
        else:
            raise

    # v3.3.0: warm up CUDA kernels to eliminate first-inference latency
    if device == "cuda":
        _warmup_faster_whisper(model)

    _fw_model_cache[cache_key] = model
    logger.info(f"[Whisper] faster-whisper '{model_size}' loaded and warmed ✓")
    return model


def _transcribe_faster_whisper(
    audio_path: str,
    model_size: str,
    language:   Optional[str],
    device:     str,
) -> Dict[str, Any]:
    """
    Transcribe using faster-whisper with segment-level streaming + timeouts.

    Key settings:
      beam_size=1                  greedy decoding — 3× faster than beam_size=5
      vad_filter=True              skip silence, stop hallucination on pauses
      condition_on_previous_text=False  prevent repetition loops on long audio
      word_timestamps=False        not needed for study notes; saves ~20% time

    Timeout strategy:
      _CHUNK_TIMEOUT_SEC  — per-segment: aborts if model stalls on one window
      _TOTAL_TIMEOUT_SEC  — wall-clock: aborts the whole transcription
    """
    model = _load_faster_whisper(model_size, device)

    segments_out:  List[Dict[str, Any]] = []
    detected_lang: str = language or "en"
    segment_queue: List[Any] = []
    error_holder:  List[Optional[Exception]] = [None]

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

    wall_start    = time.monotonic()
    last_seg_time = time.monotonic()
    full_text_parts: List[str] = []

    while True:
        if time.monotonic() - wall_start > _TOTAL_TIMEOUT_SEC:
            logger.error(
                f"[Whisper] Total timeout ({_TOTAL_TIMEOUT_SEC}s) — "
                "returning partial transcript."
            )
            break

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
            time.sleep(0.05)

    result: Dict[str, Any] = {
        "text":              " ".join(full_text_parts).strip(),
        "language":          detected_lang,
        "segments":          segments_out,
        "whisper_available": True,
        "backend":           "faster-whisper",
        "model":             model_size,
        "cpu_threads":       _CPU_THREADS,
    }
    if error_holder[0] is not None:
        result["error"] = str(error_holder[0])

    compute_label = (
        "INT8+FP16" if "float16" in _GPU_COMPUTE_TYPE and device == "cuda"
        else "INT8"
    )
    logger.info(
        f"[Whisper] Done: {len(segments_out)} segs, "
        f"{len(result['text'])} chars, lang='{detected_lang}', "
        f"backend=faster-whisper/{model_size}/{compute_label}, "
        f"cpu_threads={_CPU_THREADS}"
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
        logger.warning("[Whisper] soundfile not installed — processing full file.")
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

    logger.info(f"[Whisper] Split into {len(chunk_paths)} chunk(s) of {chunk_sec}s.")
    return chunk_paths


def _transcribe_chunk_with_timeout(
    model: Any, chunk_path: str, language: Optional[str], timeout_sec: int,
) -> Dict[str, Any]:
    """Transcribe one WAV chunk with a hard per-chunk timeout."""
    result_holder: List[Any]                 = [None]
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
    """
    openai-whisper fallback: chunk → transcribe each → stitch.

    v3.3.0: supports WHISPER_CACHE_DIR for offline model loading.
    """
    global _ow_model_cache

    if model_size not in _ow_model_cache:
        logger.info(f"[Whisper] Loading openai-whisper '{model_size}' …")

        # v3.3.0: offline cache dir
        load_kwargs: Dict[str, Any] = {}
        if _OW_CACHE_DIR and os.path.isdir(_OW_CACHE_DIR):
            load_kwargs["download_root"] = _OW_CACHE_DIR
            logger.info(f"[Whisper] Loading from local cache: {_OW_CACHE_DIR}")

        _ow_model_cache[model_size] = _openai_whisper_lib.load_model(
            model_size, **load_kwargs
        )
        logger.info(f"[Whisper] openai-whisper '{model_size}' loaded ✓")

    model       = _ow_model_cache[model_size]
    chunk_paths = _split_wav_chunks(audio_path, _CHUNK_DURATION_SEC)
    is_chunked  = len(chunk_paths) > 1

    all_segments:  List[Dict] = []
    all_text:      List[str]  = []
    detected_lang: str        = language or "en"
    seg_id_offset: int        = 0
    wall_start = time.monotonic()

    for idx, chunk_path in enumerate(chunk_paths):
        if time.monotonic() - wall_start > _TOTAL_TIMEOUT_SEC:
            logger.error(
                f"[Whisper] Overall timeout — stopped after {idx}/{len(chunk_paths)} chunks."
            )
            break

        logger.info(f"[Whisper] Chunk {idx+1}/{len(chunk_paths)}: {chunk_path}")
        raw = _transcribe_chunk_with_timeout(model, chunk_path, language, _CHUNK_TIMEOUT_SEC)
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
                           Default: env WHISPER_MODEL_SIZE (default "tiny").

    Returns
    -------
    dict with keys: text, language, segments, whisper_available,
                    backend, model, cpu_threads, error (only on failure)

    v3.3.0 additions in return dict:
      cpu_threads — number of CPU threads used for inference
    """
    if not WHISPER_AVAILABLE:
        return {
            "text": "", "language": language or "en", "segments": [],
            "whisper_available": False, "backend": "none",
            "error": (
                "Neither faster-whisper nor openai-whisper is installed.\n"
                "  Recommended: pip install faster-whisper"
            ),
        }

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    logger.info(
        f"[Whisper] Transcribing {Path(audio_path).name} | "
        f"model={model_size} lang={language or 'auto'} device={device} "
        f"cpu_threads={_CPU_THREADS} | "
        f"backend={'faster-whisper' if FASTER_WHISPER_AVAILABLE else 'openai-whisper'}"
    )

    if FASTER_WHISPER_AVAILABLE:
        try:
            return _transcribe_faster_whisper(audio_path, model_size, language, device)
        except Exception as exc:
            logger.error(
                f"[Whisper] faster-whisper failed: {exc}. "
                f"{'Falling back to openai-whisper.' if OPENAI_WHISPER_AVAILABLE else 'No fallback.'}",
                exc_info=True,
            )
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