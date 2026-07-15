"""
Multi-Modal Academic Intelligence System  v3.0.0
================================================
Transforms lecture videos, slide images, and audio recordings into
structured student learning materials.

What's new in v3
----------------
  ① Slide change detection   — frame differencing; skips duplicate slides
                               before OCR or LLM are ever called
  ② Semantic deduplication   — sentence-transformer clustering of concepts,
                               definitions, and formulas before notes/cards
  ③ Knowledge graph          — NetworkX concept graph per video
                               → GET /results/graph/{stem}  (D3.js JSON)
  ④ Multi-language           — Whisper auto-detects lecture language;
                               OCR + LLM prompts updated accordingly
  ⑤ Student progress API     — in-memory flashcard review + spaced repetition
                               (persisted to JSON on disk; no database)

Outputs per video
  ① JSON API         GET /results/video
  ② Markdown notes   GET /results/notes/{stem}
  ③ PDF report       GET /results/pdf/{stem}        ← file download
  ④ Q&A Flashcards   GET /results/flashcards/{stem}
  ⑤ MCQ Quiz         GET /results/quiz/{stem}
  ⑥ Knowledge Graph  GET /results/graph/{stem}      ← NEW

Student progress (in-memory + JSON on disk, no database)
  PATCH  /flashcards/{video_stem}/{card_index}/review   ← NEW
  GET    /progress/{video_stem}                         ← NEW
  GET    /progress/{video_stem}/due                     ← NEW

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Internal modules ──────────────────────────────────────────────────────────
from video_pipeline.config import config
from video_pipeline.core.stream_manager import StreamManager
from video_pipeline.detection.ocr import OCRExtractor
from video_pipeline.reasoning.phi3_engine import Phi3Reasoner as LlamaReasoner
from video_pipeline.utils.device import setup_device
from video_pipeline.utils.logger import get_logger

from metadata.video_metadata import extract_video_metadata

from academic_system.prompts import (
    prompt_frame_extract,
    prompt_image_extract,
    prompt_audio_topics,
    prompt_lecture_summary,
    prompt_study_notes,
    prompt_flashcards,
    prompt_quiz,
    prompt_combined_analysis,
    prompt_combined_outputs,
    prompt_everything,
)
from academic_system.pdf_generator import generate_pdf_report
from academic_system.whisper_transcriber import (
    extract_audio, transcribe, convert_to_wav, WHISPER_AVAILABLE,
)

# ── v3 additions ──────────────────────────────────────────────────────────────
from academic_system.slide_detector   import SlideChangeDetector
from academic_system.deduplicator     import SemanticDeduplicator
from academic_system.knowledge_graph  import KnowledgeGraphBuilder
from academic_system.language_support import LanguageDetector

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  APP
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Modal Academic Intelligence System",
    description=(
        "Transforms lecture videos, slide images, and audio recordings into "
        "structured student learning materials.\n\n"
        "**v3.0.0:** slide change detection · semantic deduplication · "
        "knowledge graph · multi-language · student progress API"
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
#  DIRECTORIES
# ──────────────────────────────────────────────────────────────────────────────

UPLOAD_DIR      = "uploads"
AUDIO_DIR       = "audio"
IMAGE_DIR       = "images"
FRAMES_BASE_DIR = "extracted_frames"
NOTES_DIR       = "study_notes"
PDF_DIR         = "pdf_reports"
FLASHCARD_DIR   = "flashcards"
QUIZ_DIR        = "quizzes"
GRAPH_DIR       = "knowledge_graphs"
PROGRESS_DIR    = "student_progress"   # JSON progress files per video

SERVER_HOST      = os.environ.get("SERVER_HOST",      "127.0.0.1")
SERVER_PORT      = os.environ.get("SERVER_PORT",      "8000")
SERVER_BASE_PATH = os.environ.get("SERVER_BASE_PATH", "").rstrip("/")

for _d in (UPLOAD_DIR, AUDIO_DIR, IMAGE_DIR, FRAMES_BASE_DIR,
           NOTES_DIR, PDF_DIR, FLASHCARD_DIR, QUIZ_DIR,
           GRAPH_DIR, PROGRESS_DIR, config.output_dir):
    os.makedirs(_d, exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}


# ──────────────────────────────────────────────────────────────────────────────
#  GLOBAL STATE
# ──────────────────────────────────────────────────────────────────────────────

pipeline_task:    Optional[asyncio.Task] = None
pipeline_running: bool = False

academic_results:      Dict[str, Dict[str, Any]] = {}
stream_frame_counters: Dict[str, int]            = {}

# v3 singletons
_deduplicator  = SemanticDeduplicator()
_graph_builder = KnowledgeGraphBuilder()
_lang_detector = LanguageDetector()

# Student progress store — {video_stem: {card_key: [review_events]}}
# card_key = "{video_stem}:{card_index}"
_progress: Dict[str, Dict[str, List[Dict]]] = {}


# ──────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def serialize(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def ocr_to_text(raw_ocr: Any) -> str:
    if isinstance(raw_ocr, list):
        parts = []
        for item in raw_ocr:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(p for p in parts if p).strip()
    return str(raw_ocr).strip()


def _assert_ext(filename: str, allowed: set, label: str) -> None:
    if Path(filename).suffix.lower() not in allowed:
        raise HTTPException(415, f"'{filename}' is not a supported {label} file.")


def _make_frame_url(rel: str) -> str:
    return (
        f"http://{SERVER_HOST}:{SERVER_PORT}"
        f"{SERVER_BASE_PATH}/{rel.replace(os.sep, '/')}"
    )


def _video_frames_dir(video_path: str) -> str:
    d = os.path.join(FRAMES_BASE_DIR, Path(video_path).stem)
    os.makedirs(d, exist_ok=True)
    return d


def save_frame(frame: np.ndarray, frame_id: int, timestamp: float,
               video_path: str) -> Tuple[str, str]:
    frames_dir = _video_frames_dir(video_path)
    filename   = f"frame_{frame_id:05d}_t{timestamp:.3f}s.jpg"
    path       = os.path.join(frames_dir, filename)
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    rel = os.path.relpath(path)
    return rel, _make_frame_url(rel)


def write_frames_index(video_path: str, index: List[Dict]) -> str:
    d = _video_frames_dir(video_path)
    p = os.path.join(d, "frames_index.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    return p


def write_text_file(path: str, content: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Written → {path}")
    return path


def write_json_file(path: str, data: Any) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Written → {path}")
    return path


def stem_path(video_path: str) -> str:
    return Path(video_path).stem


# ── LLM helpers — per-task token-limit overrides ─────────────────────────────
#
# The engine (phi3_engine.py v2.0.5+) now computes a safe output budget
# internally via _safe_new_tokens(), so patching max_new_tokens here is safe
# even for large values like 4096 — the engine will cap it correctly.

_PHI3_CTX = int(os.environ.get("PHI3_CONTEXT_LENGTH", "4096"))
_CTX_MARGIN = 64


def _safe_max_tokens(llm: LlamaReasoner, prompt: str, desired: int) -> int:
    """
    Return the largest output token budget that fits in the context window.

    Falls back to a character-count estimate only if the engine does not
    expose context_length (i.e. a different engine is in use).
    With phi3_engine v2.0.5+ the engine handles this internally, so the
    capping here is a secondary safety net.
    """
    ctx = int(
        getattr(llm, "context_length", None)
        or getattr(llm, "max_position_embeddings", None)
        or config.phi3_context_length
    )
    prompt_tokens_estimate = max(1, len(prompt) // 4)
    available = ctx - prompt_tokens_estimate - _CTX_MARGIN
    safe = max(64, min(desired, available))
    if safe < desired:
        logger.debug(
            f"[main] Token budget pre-capped: desired={desired} "
            f"prompt_est={prompt_tokens_estimate} ctx={ctx} -> safe={safe}"
        )
    return safe


def _llm_reason(llm: LlamaReasoner, prompt: str, max_tokens: int) -> Any:
    """
    Call llm.reason() with a per-task token limit.

    Temporarily patches llm.max_new_tokens, restores on exit.
    The engine (v2.0.5+) handles context-window safety internally,
    but _safe_max_tokens provides a secondary cap as a fallback.
    After the call, llm._last_raw_output holds the raw text for
    partial JSON recovery in the caller.
    """
    safe     = _safe_max_tokens(llm, prompt, max_tokens)
    original = getattr(llm, "max_new_tokens", None)
    try:
        if original is not None:
            llm.max_new_tokens = safe
        return llm.reason(prompt)
    finally:
        if original is not None:
            llm.max_new_tokens = original


def _llm_reason_text(llm: LlamaReasoner, prompt: str, max_tokens: int) -> str:
    """Same as _llm_reason but calls reason_text() for Markdown output."""
    safe     = _safe_max_tokens(llm, prompt, max_tokens)
    original = getattr(llm, "max_new_tokens", None)
    try:
        if original is not None:
            llm.max_new_tokens = safe
        return llm.reason_text(prompt)
    finally:
        if original is not None:
            llm.max_new_tokens = original


def _extract_first_json_object(raw: str) -> Dict:
    """
    Extract the first complete, brace-balanced JSON object from raw text.

    Handles the most common Phi-3 failure mode:
      - Model wraps output in ```json ... ``` fences
      - Model appends a comment or extra sentence after the closing brace
      - json.loads() raises "Extra data" because it sees characters past pos N

    Strategy: scan character-by-character tracking brace depth.
    The moment depth returns to 0 after opening, we have a complete object.
    Try json.loads on exactly that substring — ignore everything after it.
    """
    if not raw:
        return {}

    # Strip markdown fences first
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned.strip())

    # Find the start of the first '{'
    start = cleaned.find("{")
    if start == -1:
        return {}

    depth   = 0
    in_str  = False
    escape  = False

    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                # Found closing brace — try to parse exactly this slice
                candidate = cleaned[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    # Try with the full cleaned string as last resort
                    try:
                        obj = json.loads(cleaned)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                return {}
    return {}


def _partial_json_list(raw: str) -> List[Dict]:
    """
    Recover complete JSON objects from a truncated JSON array string.

    When the model hits its token limit mid-generation the closing }] is
    missing, causing json.loads to fail on the whole string.  This function
    extracts every complete {...} object that was fully generated and returns
    them as a list, discarding any incomplete trailing object.
    """
    if not raw or not raw.strip():
        return []

    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    # First try a clean parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for key in ("flashcards", "cards", "questions", "quiz", "items"):
                if isinstance(result.get(key), list):
                    return result[key]
        return []
    except json.JSONDecodeError:
        pass

    # Partial recovery: scan for complete {...} objects
    objects: List[Dict] = []
    depth   = 0
    in_str  = False
    escape  = False
    start   = None

    for i, ch in enumerate(cleaned):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(cleaned[start : i + 1])
                    if isinstance(obj, dict) and obj:
                        objects.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None

    if objects:
        logger.info(
            f"Partial JSON recovery: extracted {len(objects)} object(s) "
            f"from truncated output ({len(raw)} chars)"
        )
    return objects


def _find_video(video_stem: str) -> Optional[Dict[str, Any]]:
    for v in academic_results.values():
        if v.get("input_type") == "video" and Path(v["video_path"]).stem == video_stem:
            return v
    return None


# ── Student progress helpers ──────────────────────────────────────────────────

def _progress_path(video_stem: str) -> str:
    return os.path.join(PROGRESS_DIR, f"{video_stem}_progress.json")


def _load_progress(video_stem: str) -> Dict[str, List[Dict]]:
    p = _progress_path(video_stem)
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_progress(video_stem: str, data: Dict[str, List[Dict]]) -> None:
    write_json_file(_progress_path(video_stem), data)


# ──────────────────────────────────────────────────────────────────────────────
#  IMAGE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def process_image_academic(
    image_path: str,
    ocr:        OCRExtractor,
    llm:        LlamaReasoner,
    lang_info:  Optional[Dict] = None,
) -> Dict[str, Any]:
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Cannot read image file: {image_path}")

    raw_ocr  = ocr.extract(frame, config.ocr_confidence_threshold)
    ocr_text = ocr_to_text(raw_ocr)

    img_prompt = prompt_image_extract(ocr_text)
    if lang_info:
        img_prompt = _lang_detector.patch_prompt(img_prompt, lang_info)

    academic_content = serialize(llm.reason(img_prompt))

    return {
        "frame_id":         1,
        "timestamp":        0.0,
        "academic_content": academic_content,
        "ocr_text":         ocr_text,
        "ocr_raw":          serialize(raw_ocr),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  CORE ASYNC PIPELINE  (v3)
# ──────────────────────────────────────────────────────────────────────────────

async def run_academic_pipeline(video_paths: List[str]) -> None:
    """
    Full v3 pipeline — optimised for speed.

    Phase 1 — per frame (concurrent with Whisper):
      SlideChangeDetector → skip duplicates before OCR/LLM
      OCR → LLM content extraction → save JPEG
      Whisper transcription runs in a thread pool DURING Phase 1
      so audio processing overlaps with frame OCR.

    Phase 2 — per video (sequential LLM calls):
      audio topics → lecture summary → dedup → study notes
      → flashcards → quiz → knowledge graph → PDF
    """
    global pipeline_running, academic_results, stream_frame_counters

    pipeline_running      = True
    stream_frame_counters = {}

    device  = setup_device()
    sources = {f"stream_{i}": p for i, p in enumerate(video_paths)}

    import concurrent.futures
    import threading

    # ── Start StreamManager FIRST — frames begin flowing immediately ──────────
    stream_manager = StreamManager(sources, target_fps=config.fps)

    # ── Load Phi-3 + EasyOCR in a background thread ──────────────────────────
    # Previously: models loaded first (~16s), then frames started — wasted time.
    # Now: frame extraction (CPU) and model loading (CUDA) run concurrently.
    # The frame loop buffers OCR-only results until the LLM is ready,
    # then processes them inline. Zero startup dead time.
    #
    # VRAM safety: Phi-3 loads before EasyOCR inside the thread, so
    # bitsandbytes gets the full 4 GB budget during its allocation window.
    ocr_use_gpu = (
        device == "cuda"
        and os.environ.get("OCR_USE_GPU", "false").lower() == "true"
    )
    llm:           Any = None
    ocr_extractor: Any = None
    _models_ready  = threading.Event()
    _model_error:  List[Optional[Exception]] = [None]

    def _load_models() -> None:
        nonlocal llm, ocr_extractor
        try:
            logger.info("Loading Phi-3 (background)…")
            llm = LlamaReasoner(
                model_id       = config.reasoning_model_id,
                max_new_tokens = config.max_reasoning_tokens,
                device         = device,
                load_in_4bit   = config.phi3_load_in_4bit,
                adapter_path   = config.phi3_adapter_path or None,
            )
            logger.info(f"Loading EasyOCR (gpu={ocr_use_gpu})…")
            ocr_extractor = OCRExtractor(use_gpu=ocr_use_gpu)
            logger.info("Models ready ✓")
        except Exception as exc:
            _model_error[0] = exc
            logger.error(f"Model loading failed: {exc}", exc_info=True)
        finally:
            _models_ready.set()

    _model_thread = threading.Thread(target=_load_models, daemon=True, name="model_loader")
    _model_thread.start()

    # Thread pool for non-blocking JPEG saves — imwrite blocks ~5ms per frame
    _save_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="jpeg_save"
    )

    # ── Launch Whisper concurrently too ───────────────────────────────────────
    # All three now overlap: frames decode + Phi-3 loads + Whisper transcribes.
    _executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=len(video_paths),
        thread_name_prefix="whisper",
    )

    def _run_whisper(vp: str) -> Dict:
        audio_path = academic_results[vp].get("_audio_path")
        if not audio_path or not os.path.isfile(audio_path):
            return {}
        try:
            result = transcribe(audio_path, language=None, model_size=config.whisper_model_size)
            logger.info(f"[Whisper/bg] Done for {Path(vp).stem}: {len(result.get('segments',[]))} segments")
            return result
        except Exception as exc:
            logger.error(f"[Whisper/bg] Failed for {vp}: {exc}")
            return {"error": str(exc)}

    whisper_futures: Dict[str, concurrent.futures.Future] = {
        vp: _executor.submit(_run_whisper, vp)
        for vp in video_paths
    }
    logger.info("Started concurrently: frame extraction + model loading + Whisper.")

    # One SlideChangeDetector per stream — pure NumPy, no model needed
    slide_detectors: Dict[str, SlideChangeDetector] = {
        sid: SlideChangeDetector(
            hist_threshold             = config.slide_hist_threshold,
            ssim_threshold             = config.slide_ssim_threshold,
            min_seconds_between_slides = config.slide_min_seconds,
        )
        for sid in sources
    }

    stream_id_to_path: Dict[str, str] = {
        f"stream_{i}": p for i, p in enumerate(video_paths)
    }
    per_stream_frames: Dict[str, List[Dict]] = {s: [] for s in stream_id_to_path}
    frame_indices:     Dict[str, List[Dict]] = {vp: [] for vp in video_paths}

    slide_stats: Dict[str, Dict] = {
        sid: {"frames_seen": 0, "slides_accepted": 0, "frames_skipped": 0}
        for sid in sources
    }

    _completed_normally = False

    # ── PHASE 1: frame-level ──────────────────────────────────────────────────
    # OCR batch size — process this many accepted frames per EasyOCR call.
    # Larger batches amortise the GPU upload overhead but add slight latency.
    # 4 is the sweet spot on RTX 3050 (4 GB VRAM with Phi-3 already loaded).
    _OCR_BATCH = int(os.environ.get("OCR_BATCH_SIZE", "4"))

    # Pending OCR buffer: list of (frame, frame_id, timestamp, stream_id, video_path)
    # We accumulate _OCR_BATCH accepted frames then flush them all at once.
    _pending_ocr: List[tuple] = []

    def _flush_ocr_batch(batch: List[tuple]) -> None:
        """
        Run OCR + LLM on a batch of accepted frames.
        Called when the batch is full or the stream is exhausted.
        """
        if not batch:
            return

        # Separate frames that need OCR from those skipped due to models not ready
        if not _models_ready.is_set():
            # Models still loading — store all as low importance, no OCR
            for frame, frame_id, timestamp, stream_id, video_path, rel_path, frame_url in batch:
                _store_frame(frame_id, timestamp, stream_id, video_path,
                             rel_path, frame_url, "", {"importance": "low"})
            return

        if _model_error[0]:
            raise RuntimeError(f"Model loading failed: {_model_error[0]}")

        # Run batch OCR — single EasyOCR call for all frames
        raw_frames = [item[0] for item in batch]
        try:
            all_ocr = ocr_extractor.batch_extract(raw_frames, config.ocr_confidence_threshold)
        except Exception as exc:
            logger.warning(f"Batch OCR failed ({exc}), falling back to sequential.")
            all_ocr = [ocr_extractor.extract(f, config.ocr_confidence_threshold)
                       for f in raw_frames]

        for (frame, frame_id, timestamp, stream_id, video_path, rel_path, frame_url), raw_ocr \
                in zip(batch, all_ocr):
            ocr_text = ocr_to_text(raw_ocr)
            word_char_count = len(re.findall(r'[A-Za-z0-9]', ocr_text))

            if word_char_count < config.min_ocr_word_chars:
                academic_content = {"importance": "low"}
            else:
                try:
                    frame_token_limit = getattr(config, "max_tokens_frame", 400)
                    academic_content = serialize(
                        _llm_reason(llm,
                            prompt_frame_extract(ocr_text, timestamp, frame_id),
                            frame_token_limit,
                        )
                    )
                except Exception as exc:
                    logger.warning(f"Frame {frame_id} LLM failed: {exc}")
                    academic_content = {}

            _store_frame(frame_id, timestamp, stream_id, video_path,
                         rel_path, frame_url, ocr_text, academic_content)

    def _store_frame(frame_id, timestamp, stream_id, video_path,
                     rel_path, frame_url, ocr_text, academic_content) -> None:
        """Write a completed frame record into the results store."""
        frame_record: Dict[str, Any] = {
            "frame_id":         frame_id,
            "timestamp":        timestamp,
            "frame_path":       rel_path,
            "frame_url":        frame_url,
            "academic_content": academic_content,
            "ocr_text":         ocr_text,
        }
        vr = academic_results[video_path]
        if len(vr["per_frame_details"]) < config.max_frames_in_memory:
            vr["per_frame_details"].append(frame_record)
        per_stream_frames[stream_id].append(frame_record)

    try:
        while pipeline_running:
            stream_ids, frames, timestamps = await stream_manager.get_batch()
            if not frames:
                _completed_normally = True
                logger.info("All streams exhausted — phase 1 complete.")
                break

            for i, stream_id in enumerate(stream_ids):
                frame      = frames[i]
                timestamp  = float(timestamps[i])
                video_path = stream_id_to_path[stream_id]
                stats      = slide_stats[stream_id]

                stats["frames_seen"] += 1

                # Slide change detection — runs immediately, no model needed
                if not slide_detectors[stream_id].is_new_slide(frame, timestamp):
                    stats["frames_skipped"] += 1
                    logger.debug(f"[{stream_id}] @{timestamp:.1f}s — duplicate, skipped.")
                    continue

                stats["slides_accepted"] += 1
                stream_frame_counters[stream_id] = (
                    stream_frame_counters.get(stream_id, 0) + 1
                )
                frame_id = stream_frame_counters[stream_id]

                # Async JPEG save — fire and forget
                try:
                    _save_executor.submit(
                        lambda f=frame, fi=frame_id, ts=timestamp, vp=video_path:
                            save_frame(f, fi, ts, vp)
                    )
                    frames_dir = _video_frames_dir(video_path)
                    filename   = f"frame_{frame_id:05d}_t{timestamp:.3f}s.jpg"
                    rel_path   = os.path.relpath(os.path.join(frames_dir, filename))
                    frame_url  = _make_frame_url(rel_path)
                except Exception as exc:
                    logger.warning(f"Frame save failed: {exc}")
                    rel_path, frame_url = "", ""

                frame_indices[video_path].append({
                    "frame_id":   frame_id,
                    "stream_id":  stream_id,
                    "video_path": video_path,
                    "frame_path": rel_path,
                    "frame_url":  frame_url,
                    "timestamp":  round(timestamp, 3),
                })
                # Batched index writes — every 10 frames
                if frame_id % 10 == 0:
                    try:
                        write_frames_index(video_path, frame_indices[video_path])
                    except Exception as exc:
                        logger.warning(f"Frame index write: {exc}")

                # Queue frame for batched OCR
                _pending_ocr.append((frame, frame_id, timestamp, stream_id,
                                     video_path, rel_path, frame_url))

                # Flush when batch is full
                if len(_pending_ocr) >= _OCR_BATCH:
                    _flush_ocr_batch(_pending_ocr)
                    _pending_ocr.clear()
                    # Yield to event loop so HTTP status polls stay responsive
                    await asyncio.sleep(0)

    finally:
        # Flush any remaining frames that didn't fill a full batch
        if _pending_ocr:
            _flush_ocr_batch(_pending_ocr)
            _pending_ocr.clear()

        stream_manager.release_all()
        if not _completed_normally:
            logger.info("Pipeline stopped early — generating outputs from collected frames.")

        # Log and store slide detection stats
        for sid, stats in slide_stats.items():
            vp  = stream_id_to_path.get(sid)
            pct = 100 * stats["frames_skipped"] / max(stats["frames_seen"], 1)
            logger.info(
                f"[{sid}] Slide detection: "
                f"{stats['slides_accepted']} unique slides accepted, "
                f"{stats['frames_skipped']} duplicate frames skipped ({pct:.0f}% compute saved)"
            )
            if vp and vp in academic_results:
                academic_results[vp]["slide_change_stats"] = stats

        # Flush final frame indices
        for vp, idx in frame_indices.items():
            if idx:
                try:
                    w = write_frames_index(vp, idx)
                    if vp in academic_results:
                        academic_results[vp]["frames_index_path"] = w
                        academic_results[vp]["frames_index"]      = idx
                except Exception as exc:
                    logger.error(f"Final index write failed: {exc}")

        # ── Collect Whisper results (they ran concurrently with Phase 1) ──────
        whisper_results: Dict[str, Dict] = {}
        for vp, future in whisper_futures.items():
            try:
                whisper_results[vp] = future.result(timeout=300)
            except Exception as exc:
                logger.error(f"Whisper future failed for {vp}: {exc}")
                whisper_results[vp] = {}
        _executor.shutdown(wait=False)
        _save_executor.shutdown(wait=True)   # wait for pending JPEGs to finish

        # Ensure models finished loading before Phase 2 LLM calls.
        # (If no LLM-eligible frames appeared in Phase 1 the wait never triggered there.)
        if not _models_ready.is_set():
            logger.info("Waiting for model loading to complete before Phase 2…")
            _models_ready.wait()
        if _model_error[0]:
            logger.error(f"Model load failed — Phase 2 will be empty: {_model_error[0]}")

        # ── PHASE 2: per-video outputs ────────────────────────────────────────
        for stream_id, frames_list in per_stream_frames.items():
            video_path = stream_id_to_path[stream_id]
            vr         = academic_results[video_path]
            stem       = stem_path(video_path)

            logger.info(f"Phase 2 — generating academic outputs for: {stem}")

            if not frames_list:
                vr.update({
                    "total_frames_analysed": 0,
                    "lecture_summary": {}, "audio_topics": {},
                    "study_notes": None,   "flashcards": [],
                    "quiz": [],            "pdf_report_path": None,
                    "knowledge_graph": None,
                })
                logger.warning(f"No frames collected for {video_path}.")
                continue

            # ── 2a. Use pre-collected Whisper result (already done) ───────────
            vr.pop("_audio_path", None)
            transcription = whisper_results.get(video_path, {})
            transcript    = transcription.get("text", "")
            lang_info     = _lang_detector.from_code("en")

            if transcription and not transcription.get("error"):
                vr["audio_analysis"] = transcription
                lang_info = _lang_detector.from_whisper(transcription)
                vr["detected_language"] = {
                    "code": lang_info["code"],
                    "name": lang_info["name"],
                    "rtl":  lang_info["rtl"],
                }
                logger.info(f"Language: {lang_info['name']} (code={lang_info['code']})")
                ocr_extractor.set_languages(lang_info["ocr_langs"])
            else:
                vr["audio_analysis"] = transcription or None

            def _patch(prompt: str) -> str:
                return _lang_detector.patch_prompt(prompt, lang_info)

            # ── THE ONE CALL: everything in a single Phi-3 generation ─────────
            # Replaces 5 separate LLM calls with 1.
            # Expected time: ~90-120s instead of ~6 min.
            logger.info("Phase 2: single combined call (analysis + notes + flashcards + quiz)…")

            audio_topics:    Dict[str, Any] = {}
            lecture_summary: Dict[str, Any] = {}
            notes_md  = ""
            flashcards: List[Dict] = []
            quiz:       List[Dict] = []

            everything_prompt = _patch(prompt_everything(
                video_path, frames_list, transcript,
                sample_n=config.summary_sample_frames,
            ))

            result = serialize(
                _llm_reason(llm, everything_prompt, getattr(config, "max_tokens_everything", 1400))
            ) or {}

            # ── Robust recovery from the raw output ───────────────────────────
            # The engine's json.loads fails on "Extra data" when the model adds
            # trailing text after the closing brace (e.g. ``` or a comment).
            # _extract_first_json_object() finds the first brace-balanced {...}
            # and ignores everything after it.
            if not result:
                raw_text = getattr(llm, "_last_raw_output", "") or ""
                if raw_text:
                    result = _extract_first_json_object(raw_text) or {}
                    if result:
                        logger.info(
                            f"JSON recovery succeeded ({len(raw_text)} chars raw → "
                            f"{len(result)} top-level keys extracted)."
                        )

            if result:
                # ── Extract lecture analysis fields ───────────────────────────
                lecture_summary = {
                    "lecture_title":     result.get("lecture_title", ""),
                    "subject_area":      result.get("subject_area", ""),
                    "main_topics":       result.get("topics", []),
                    "learning_outcomes": result.get("learning_outcomes", []),
                    "summary":           result.get("summary", ""),
                    "difficulty_level":  result.get("difficulty", ""),
                }
                audio_topics = {
                    "lecture_title":    result.get("lecture_title", ""),
                    "subject_area":     result.get("subject_area", ""),
                    "topics_covered":   result.get("topics", []),
                    "key_concepts":     [
                        {"concept": c, "explanation": ""}
                        for c in result.get("key_concepts", [])
                    ],
                    "important_points": result.get("learning_outcomes", []),
                    "summary":          result.get("summary", ""),
                }

                # ── Extract study notes ───────────────────────────────────────
                raw_notes = result.get("study_notes", "")
                if raw_notes:
                    notes_md = raw_notes.replace("\\n", "\n").replace("\\t", "\t")
                    title = lecture_summary.get("lecture_title", stem)
                    if not notes_md.lstrip().startswith("#"):
                        notes_md = f"# Study Notes: {title}\n\n" + notes_md
                else:
                    # study_notes was truncated or empty — build from other fields
                    title    = lecture_summary.get("lecture_title", stem)
                    summary  = result.get("summary", "")
                    topics   = result.get("topics", [])
                    concepts = result.get("key_concepts", [])
                    outcomes = result.get("learning_outcomes", [])
                    lines = [f"# Study Notes: {title}", ""]
                    if summary:
                        lines += ["## Overview", "", summary, ""]
                    if topics:
                        lines += ["## Topics Covered", ""] + [f"- {t}" for t in topics] + [""]
                    if concepts:
                        lines += ["## Key Concepts", ""] + [f"- {c}" for c in concepts] + [""]
                    if outcomes:
                        lines += ["## Learning Outcomes", ""] + [f"- {o}" for o in outcomes] + [""]
                    notes_md = "\n".join(lines)
                    logger.info("study_notes field empty — built from metadata fields.")

                # ── Extract flashcards ────────────────────────────────────────
                raw_fc = result.get("flashcards", [])
                for card in raw_fc:
                    if isinstance(card, dict):
                        # Normalise compact schema {q, a, topic} → {question, answer, topic}
                        flashcards.append({
                            "question":   card.get("question") or card.get("q", ""),
                            "answer":     card.get("answer")   or card.get("a", ""),
                            "topic":      card.get("topic", ""),
                            "difficulty": card.get("difficulty", "medium"),
                        })

                # ── Extract quiz ──────────────────────────────────────────────
                raw_qz = result.get("quiz", [])
                for q in raw_qz:
                    if isinstance(q, dict):
                        # Normalise compact schema {q, A, B, C, D, ans, why}
                        opts = {}
                        for k in ("A", "B", "C", "D"):
                            if k in q:
                                opts[k] = q[k]
                        if not opts and "options" in q:
                            opts = q["options"]
                        quiz.append({
                            "question":       q.get("question") or q.get("q", ""),
                            "options":        opts,
                            "correct_answer": q.get("correct_answer") or q.get("ans", ""),
                            "explanation":    q.get("explanation")    or q.get("why", ""),
                            "topic":          q.get("topic", ""),
                        })

                logger.info(
                    f"Single-call done: title='{lecture_summary.get('lecture_title','?')[:40]}' "
                    f"notes={len(notes_md)} chars  "
                    f"flashcards={len(flashcards)}  quiz={len(quiz)}"
                )
            else:
                logger.error("Single-call returned empty result — falling back to 3-call pipeline.")
                # ── Fallback: 3 separate calls ────────────────────────────────
                try:
                    combined = serialize(_llm_reason(
                        llm,
                        _patch(prompt_combined_analysis(
                            video_path, frames_list, transcript,
                            sample_n=config.summary_sample_frames,
                        )),
                        config.max_tokens_summary * 2,
                    )) or {}
                    audio_topics    = combined.get("audio_analysis", {}) or {}
                    lecture_summary = combined.get("lecture_summary", {}) or {}
                    vr["audio_topics"]    = audio_topics
                    vr["lecture_summary"] = lecture_summary
                except Exception as e:
                    logger.error(f"Fallback combined analysis failed: {e}")

                try:
                    notes_md = _llm_reason_text(
                        llm,
                        _patch(prompt_study_notes(
                            video_path, frames_list, transcript,
                            audio_topics, lecture_summary,
                            sample_n=config.notes_sample_frames,
                        )),
                        config.max_tokens_notes,
                    )
                except Exception as e:
                    logger.error(f"Fallback study notes failed: {e}")
                    notes_md = "Study notes generation failed."

                try:
                    co = serialize(_llm_reason(
                        llm,
                        _patch(prompt_combined_outputs(
                            video_path, frames_list, audio_topics, lecture_summary,
                        )),
                        config.max_tokens_flashcards,
                    )) or {}
                    flashcards = co.get("flashcards", [])
                    quiz       = co.get("quiz", [])
                except Exception as e:
                    logger.error(f"Fallback combined outputs failed: {e}")

            vr["audio_topics"]          = audio_topics
            vr["lecture_summary"]       = lecture_summary
            vr["total_frames_analysed"] = len(frames_list)

            # ── Deduplication — run all 3 in parallel threads ─────────────────
            # Each is pure CPU/NumPy with no shared state — perfectly thread-safe.
            audio_concept_frames: List[Dict] = []
            for c in audio_topics.get("key_concepts", []):
                name = (c.get("concept") or c.get("name") or "") if isinstance(c, dict) else str(c)
                if name.strip():
                    audio_concept_frames.append(
                        {"academic_content": {"key_concepts": [name.strip()]}}
                    )
            all_frames_for_dedup = frames_list + audio_concept_frames

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as _dedup_pool:
                f_concepts = _dedup_pool.submit(_deduplicator.deduplicate_concepts, all_frames_for_dedup)
                f_defs     = _dedup_pool.submit(_deduplicator.deduplicate_definitions, frames_list)
                f_formulas = _dedup_pool.submit(_deduplicator.deduplicate_formulas, frames_list)
                deduped_concepts = f_concepts.result()
                deduped_defs     = f_defs.result()
                deduped_formulas = f_formulas.result()

            vr["deduped_concepts"] = deduped_concepts
            vr["deduped_formulas"] = deduped_formulas
            logger.info(
                f"Dedup ({_deduplicator.backend}): "
                f"{len(deduped_concepts)} concepts, {len(deduped_formulas)} formulas."
            )

            # ── Persist outputs ───────────────────────────────────────────────
            if not notes_md:
                notes_md = f"# Study Notes: {stem}\n\nNotes generation failed."
            notes_path = write_text_file(
                os.path.join(NOTES_DIR, f"{stem}_study_notes.md"), notes_md
            )
            vr["study_notes"]      = notes_md
            vr["study_notes_path"] = notes_path

            flash_path = write_json_file(
                os.path.join(FLASHCARD_DIR, f"{stem}_flashcards.json"), flashcards
            )
            vr["flashcards"]      = flashcards
            vr["flashcards_path"] = flash_path

            quiz_path = write_json_file(
                os.path.join(QUIZ_DIR, f"{stem}_quiz.json"), quiz
            )
            vr["quiz"]      = quiz
            vr["quiz_path"] = quiz_path

            # ── PDF report + knowledge graph — run concurrently ───────────────
            # Neither depends on the other. Both are CPU-bound (~2s each).
            def _build_graph():
                try:
                    graph    = _graph_builder.build(frames_list, audio_topics, lecture_summary)
                    graph_d3 = _graph_builder.to_d3_json(graph)
                    graph_path = _graph_builder.save(
                        graph, os.path.join(GRAPH_DIR, f"{stem}_knowledge_graph.json")
                    )
                    vr["knowledge_graph"]      = graph_d3
                    vr["knowledge_graph_path"] = graph_path
                    api_data = _graph_builder.to_api_json(graph)
                    logger.info(
                        f"Knowledge graph: {api_data.get('num_nodes', 0)} nodes, "
                        f"{api_data.get('num_edges', 0)} edges."
                    )
                except Exception as exc:
                    logger.error(f"Knowledge graph failed: {exc}")
                    vr["knowledge_graph"] = None

            def _build_pdf():
                try:
                    pdf_path = generate_pdf_report(
                        video_path      = video_path,
                        pdf_dir         = PDF_DIR,
                        lecture_summary = lecture_summary,
                        audio_topics    = audio_topics,
                        frame_analyses  = frames_list,
                        flashcards      = flashcards,
                        transcript_text = transcript,
                    )
                    vr["pdf_report_path"] = pdf_path
                    logger.info(f"PDF report: {pdf_path}")
                except Exception as exc:
                    logger.error(f"PDF generation failed: {exc}")
                    vr["pdf_report_path"] = None

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _output_pool:
                _gf = _output_pool.submit(_build_graph)
                _pf = _output_pool.submit(_build_pdf)
                _gf.result()
                _pf.result()

        pipeline_running = False
        logger.info("Academic pipeline v3 complete.")


# ──────────────────────────────────────────────────────────────────────────────
#  PYDANTIC MODELS
# ──────────────────────────────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    confidence: int  = Field(..., ge=1, le=5, description="1=very hard → 5=very easy")
    correct:    bool = Field(False, description="Did the student answer correctly?")
    session_id: Optional[str] = Field(None, description="Optional study session ID")


# ──────────────────────────────────────────────────────────────────────────────
#  UPLOAD ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/upload/video", summary="Upload 1–3 lecture videos", tags=["Upload"])
async def upload_video(
    file1: UploadFile = File(None),
    file2: UploadFile = File(None),
    file3: UploadFile = File(None),
) -> JSONResponse:
    global pipeline_task, pipeline_running

    uploaded = [f for f in (file1, file2, file3) if f is not None]
    if not uploaded:
        raise HTTPException(400, "At least one video file is required.")
    for f in uploaded:
        _assert_ext(f.filename, VIDEO_EXTENSIONS, "video")

    if pipeline_task and not pipeline_task.done():
        pipeline_running = False
        try:
            await asyncio.wait_for(pipeline_task, timeout=5.0)
        except asyncio.TimeoutError:
            pipeline_task.cancel()

    for key in [k for k, v in academic_results.items() if v.get("input_type") == "video"]:
        del academic_results[key]

    video_paths: List[str] = []

    for upload in uploaded:
        dest = os.path.join(UPLOAD_DIR, upload.filename)
        with open(dest, "wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        logger.info(f"Video saved → {dest}")

        meta       = extract_video_metadata(dest)
        audio_path = extract_audio(dest, AUDIO_DIR)
        frames_dir = _video_frames_dir(dest)

        academic_results[dest] = {
            "input_type":            "video",
            "video_path":            dest,
            "metadata":              meta,
            "_audio_path":           audio_path,
            "frames_dir":            frames_dir,
            "frames_index_path":     os.path.join(frames_dir, "frames_index.json"),
            "frames_index":          [],
            "total_frames_analysed": 0,
            "per_frame_details":     [],
            "audio_analysis":        None,
            "audio_topics":          {},
            "lecture_summary":       {},
            "detected_language":     None,
            "deduped_concepts":      [],
            "deduped_formulas":      [],
            "study_notes":           None,
            "flashcards":            [],
            "quiz":                  [],
            "knowledge_graph":       None,
            "pdf_report_path":       None,
            "slide_change_stats":    {},
        }
        video_paths.append(dest)

    pipeline_task = asyncio.create_task(run_academic_pipeline(video_paths))

    return JSONResponse({
        "message": f"Academic pipeline v3 started for {len(video_paths)} video(s).",
        "videos":  video_paths,
        "poll":    "GET /status",
        "outputs": {
            "json":            "GET /results/video",
            "study_notes":     "GET /results/notes/{stem}",
            "pdf":             "GET /results/pdf/{stem}",
            "flashcards":      "GET /results/flashcards/{stem}",
            "quiz":            "GET /results/quiz/{stem}",
            "knowledge_graph": "GET /results/graph/{stem}",
            "frames":          "GET /results/frames/{stem}",
        },
        "v3_features": [
            "slide change detection — duplicates skipped before OCR/LLM",
            "semantic deduplication — unique concepts fed to notes + flashcards",
            "Whisper language auto-detection — OCR + LLM prompts updated",
            "knowledge graph — GET /results/graph/{stem}",
        ],
    })


@app.post("/upload/image", summary="Upload 1–3 slide or diagram images", tags=["Upload"])
async def upload_image(
    file1:    UploadFile = File(None),
    file2:    UploadFile = File(None),
    file3:    UploadFile = File(None),
    language: str        = Query(
        "en",
        description="ISO 639-1 language code for OCR (e.g. hi, zh, ar). GET /languages for full list."
    ),
) -> JSONResponse:
    uploaded = [f for f in (file1, file2, file3) if f is not None]
    if not uploaded:
        raise HTTPException(400, "At least one image file is required.")
    for f in uploaded:
        _assert_ext(f.filename, IMAGE_EXTENSIONS, "image")

    lang_info = _lang_detector.from_code(language)
    device    = setup_device()
    ocr       = OCRExtractor(use_gpu=(device == "cuda"), languages=lang_info["ocr_langs"])
    llm       = LlamaReasoner(
        model_id       = config.reasoning_model_id,
        max_new_tokens = config.max_reasoning_tokens,
        device         = device,
        load_in_4bit   = config.phi3_load_in_4bit,
        adapter_path   = config.phi3_adapter_path or None,
    )

    results = []
    for upload in uploaded:
        dest = os.path.join(IMAGE_DIR, upload.filename)
        with open(dest, "wb") as fh:
            shutil.copyfileobj(upload.file, fh)

        try:
            analysis = process_image_academic(dest, ocr, llm, lang_info)
            entry    = {
                "input_type": "image",
                "image_path": dest,
                "language":   lang_info["name"],
                "analysis":   analysis,
            }
        except Exception as exc:
            logger.error(f"Image processing failed: {exc}")
            entry = {"input_type": "image", "image_path": dest, "error": str(exc)}

        academic_results[dest] = entry
        results.append(entry)

    return JSONResponse({
        "message":  f"{len(results)} image(s) analysed.",
        "language": lang_info["name"],
        "results":  results,
    })


@app.post("/upload/audio", summary="Upload 1–3 lecture audio files", tags=["Upload"])
async def upload_audio(
    file1: UploadFile = File(None),
    file2: UploadFile = File(None),
    file3: UploadFile = File(None),
) -> JSONResponse:
    uploaded = [f for f in (file1, file2, file3) if f is not None]
    if not uploaded:
        raise HTTPException(400, "At least one audio file is required.")
    for f in uploaded:
        _assert_ext(f.filename, AUDIO_EXTENSIONS, "audio")

    results = []
    for upload in uploaded:
        dest = os.path.join(AUDIO_DIR, upload.filename)
        with open(dest, "wb") as fh:
            shutil.copyfileobj(upload.file, fh)

        ext = Path(dest).suffix.lower()
        processing_path = convert_to_wav(dest, AUDIO_DIR) if ext != ".wav" else dest

        try:
            transcription = transcribe(
                processing_path,
                language   = None,   # auto-detect
                model_size = config.whisper_model_size,
            )
            lang_info = _lang_detector.from_whisper(transcription)
            entry = {
                "input_type":        "audio",
                "audio_path":        dest,
                "detected_language": {"code": lang_info["code"], "name": lang_info["name"]},
                "transcription":     transcription,
            }
        except Exception as exc:
            logger.error(f"Audio processing failed: {exc}")
            entry = {"input_type": "audio", "audio_path": dest, "error": str(exc)}

        academic_results[dest] = entry
        results.append(entry)

    return JSONResponse({
        "message":           f"{len(results)} audio file(s) transcribed.",
        "results":           results,
        "whisper_available": WHISPER_AVAILABLE,
        "note":              "Language is auto-detected by Whisper.",
    })


# ──────────────────────────────────────────────────────────────────────────────
#  CONTROL
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/stop", summary="Stop the running pipeline", tags=["Control"])
async def stop_pipeline() -> JSONResponse:
    global pipeline_running
    pipeline_running = False
    return JSONResponse({"message": "Stop signal sent. Outputs generated from collected frames."})


@app.delete("/results", summary="Clear all in-memory results", tags=["Control"])
async def clear_results() -> JSONResponse:
    academic_results.clear()
    return JSONResponse({"message": "All in-memory results cleared."})


# ──────────────────────────────────────────────────────────────────────────────
#  STATUS
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/status", summary="Pipeline progress and per-video readiness", tags=["Status"])
async def status() -> JSONResponse:
    videos = {p: v for p, v in academic_results.items() if v.get("input_type") == "video"}
    images = {p: v for p, v in academic_results.items() if v.get("input_type") == "image"}
    audios = {p: v for p, v in academic_results.items() if v.get("input_type") == "audio"}

    video_states = {}
    for p, v in videos.items():
        ss   = v.get("slide_change_stats", {})
        lang = v.get("detected_language") or {}
        video_states[os.path.basename(p)] = {
            "frames_seen":              ss.get("frames_seen", 0),
            "unique_slides_accepted":   ss.get("slides_accepted", len(v["per_frame_details"])),
            "duplicate_frames_skipped": ss.get("frames_skipped", 0),
            "frames_collected":         len(v["per_frame_details"]),
            "detected_language":        lang.get("name"),
            "language_code":            lang.get("code"),
            "deduped_concepts":         len(v.get("deduped_concepts", [])),
            "deduped_formulas":         len(v.get("deduped_formulas", [])),
            "audio_ready":              v.get("audio_analysis") is not None,
            "summary_ready":            bool(v.get("lecture_summary")),
            "study_notes_ready":        v.get("study_notes") is not None,
            "flashcards_ready":         bool(v.get("flashcards")),
            "flashcard_count":          len(v.get("flashcards", [])),
            "quiz_ready":               bool(v.get("quiz")),
            "quiz_count":               len(v.get("quiz", [])),
            "graph_ready":              v.get("knowledge_graph") is not None,
            "pdf_ready":                v.get("pdf_report_path") is not None,
            "lecture_title":            v.get("lecture_summary", {}).get("lecture_title"),
            "subject_area":             v.get("lecture_summary", {}).get("subject_area"),
            "difficulty":               v.get("lecture_summary", {}).get("difficulty_level"),
        }

    return JSONResponse({
        "pipeline_running":       pipeline_running,
        "task_done":              pipeline_task.done() if pipeline_task else True,
        "total_frames_collected": sum(len(v["per_frame_details"]) for v in videos.values()),
        "videos_in_pipeline":     len(videos),
        "images_analysed":        len(images),
        "audio_files_analysed":   len(audios),
        "videos":                 video_states,
    })


# ──────────────────────────────────────────────────────────────────────────────
#  RESULTS
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/results/video",  summary="Full JSON results for all videos",    tags=["Results"])
async def results_video() -> List[Dict[str, Any]]:
    return [v for v in academic_results.values() if v.get("input_type") == "video"]

@app.get("/results/image",  summary="Analysis results for all images",     tags=["Results"])
async def results_image() -> List[Dict[str, Any]]:
    return [v for v in academic_results.values() if v.get("input_type") == "image"]

@app.get("/results/audio",  summary="Transcription results for all audio", tags=["Results"])
async def results_audio() -> List[Dict[str, Any]]:
    return [v for v in academic_results.values() if v.get("input_type") == "audio"]


@app.get("/results/notes/{video_stem}", summary="Markdown study notes", tags=["Results"])
async def results_notes(video_stem: str) -> PlainTextResponse:
    v = _find_video(video_stem)
    if v:
        notes = v.get("study_notes")
        if notes:
            return PlainTextResponse(notes, media_type="text/markdown")
        raise HTTPException(503, "Study notes not yet generated — check /status.")
    p = os.path.join(NOTES_DIR, f"{video_stem}_study_notes.md")
    if os.path.isfile(p):
        return PlainTextResponse(open(p, encoding="utf-8").read(), media_type="text/markdown")
    raise HTTPException(404, f"No study notes found for '{video_stem}'.")


@app.get("/results/pdf/{video_stem}", summary="Download PDF academic report", tags=["Results"])
async def results_pdf(video_stem: str) -> FileResponse:
    v = _find_video(video_stem)
    if v:
        pdf_path = v.get("pdf_report_path")
        if pdf_path and os.path.isfile(pdf_path):
            return FileResponse(
                pdf_path, media_type="application/pdf",
                filename=os.path.basename(pdf_path),
            )
        raise HTTPException(503, "PDF not yet generated — check /status.")
    p = os.path.join(PDF_DIR, f"{video_stem}_academic_report.pdf")
    if os.path.isfile(p):
        return FileResponse(p, media_type="application/pdf", filename=os.path.basename(p))
    raise HTTPException(404, f"No PDF found for '{video_stem}'.")


@app.get("/results/flashcards/{video_stem}", summary="Q&A flashcards", tags=["Results"])
async def results_flashcards(video_stem: str) -> List[Dict]:
    v = _find_video(video_stem)
    if v:
        cards = v.get("flashcards")
        if cards is not None:
            return cards
        raise HTTPException(503, "Flashcards not yet generated — check /status.")
    p = os.path.join(FLASHCARD_DIR, f"{video_stem}_flashcards.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(404, f"No flashcards found for '{video_stem}'.")


@app.get("/results/quiz/{video_stem}", summary="MCQ quiz", tags=["Results"])
async def results_quiz(video_stem: str) -> List[Dict]:
    v = _find_video(video_stem)
    if v:
        quiz = v.get("quiz")
        if quiz is not None:
            return quiz
        raise HTTPException(503, "Quiz not yet generated — check /status.")
    p = os.path.join(QUIZ_DIR, f"{video_stem}_quiz.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(404, f"No quiz found for '{video_stem}'.")


@app.get("/results/graph/{video_stem}",
         summary="Knowledge graph in D3.js format", tags=["Results"])
async def results_graph(video_stem: str) -> Dict:
    """
    Returns the concept co-occurrence graph for a processed video.

    Format is D3.js force-directed graph::

        {
          "nodes": [{"id": "newton's second law", "label": "Newton's Second Law",
                     "type": "concept", "freq": 5, "index": 0}, ...],
          "links": [{"source": 0, "target": 3, "relation": "co_occurs", "weight": 2}, ...]
        }

    Drop into a D3 force simulation::

        d3.json("/results/graph/my_lecture").then(data => {
          const sim = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.index))
            .force("charge", d3.forceManyBody())
            .force("center", d3.forceCenter(width/2, height/2));
        });
    """
    v = _find_video(video_stem)
    if v:
        kg = v.get("knowledge_graph")
        if kg is not None:
            return kg
        raise HTTPException(503, "Knowledge graph not yet generated — check /status.")
    p = os.path.join(GRAPH_DIR, f"{video_stem}_knowledge_graph.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(404, f"No knowledge graph found for '{video_stem}'.")


@app.get("/results/frames/{video_stem}", summary="Frame index", tags=["Results"])
async def results_frames(video_stem: str) -> List[Dict]:
    v = _find_video(video_stem)
    if v:
        return v.get("frames_index", [])
    p = os.path.join(FRAMES_BASE_DIR, video_stem, "frames_index.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(404, f"No frame index found for '{video_stem}'.")


@app.get("/results/latest", summary="Most recent N frames", tags=["Results"])
async def latest(n: int = 10) -> List[Dict]:
    all_frames = [
        {**fr, "video_path": v["video_path"]}
        for v in academic_results.values()
        if v.get("input_type") == "video"
        for fr in v["per_frame_details"]
    ]
    return all_frames[-n:]


# ──────────────────────────────────────────────────────────────────────────────
#  STUDENT PROGRESS API  (in-memory + JSON on disk)
# ──────────────────────────────────────────────────────────────────────────────

@app.patch(
    "/flashcards/{video_stem}/{card_index}/review",
    summary="Record a flashcard review (confidence 1–5)",
    tags=["Student Progress"],
)
async def review_flashcard(
    video_stem: str,
    card_index: int,
    body:       ReviewRequest,
) -> JSONResponse:
    """
    Record that a student reviewed a specific flashcard.

    | Field | Type | Description |
    |-------|------|-------------|
    | confidence | int 1–5 | 1=very hard, 5=very easy |
    | correct | bool | Did the student get it right? |
    | session_id | str (optional) | Group reviews by study session |

    Cards with average confidence ≤ 3 resurface in `GET /progress/{stem}/due`.
    Progress is saved to `student_progress/{stem}_progress.json`.
    """
    global _progress

    # Validate card exists
    v = _find_video(video_stem)
    if v:
        flashcards = v.get("flashcards", [])
        if card_index < 0 or card_index >= len(flashcards):
            raise HTTPException(
                404,
                f"Card index {card_index} out of range "
                f"(0–{len(flashcards)-1} for this video)."
            )
    else:
        p = os.path.join(FLASHCARD_DIR, f"{video_stem}_flashcards.json")
        if not os.path.isfile(p):
            raise HTTPException(404, f"No flashcards found for '{video_stem}'.")

    card_key = f"{video_stem}:{card_index}"

    if video_stem not in _progress:
        _progress[video_stem] = _load_progress(video_stem)
    if card_key not in _progress[video_stem]:
        _progress[video_stem][card_key] = []

    event = {
        "card_index":  card_index,
        "confidence":  body.confidence,
        "correct":     body.correct,
        "session_id":  body.session_id,
        "reviewed_at": time.time(),
    }
    _progress[video_stem][card_key].append(event)
    _save_progress(video_stem, _progress[video_stem])

    reviews  = _progress[video_stem][card_key]
    avg_conf = sum(r["confidence"] for r in reviews) / len(reviews)

    return JSONResponse({
        "message":        "Review recorded.",
        "card_key":       card_key,
        "total_reviews":  len(reviews),
        "avg_confidence": round(avg_conf, 2),
        "correct_count":  sum(1 for r in reviews if r["correct"]),
        "needs_review":   avg_conf <= 3.0,
    })


@app.get(
    "/progress/{video_stem}",
    summary="Review history for all flashcards of a video",
    tags=["Student Progress"],
)
async def get_progress(
    video_stem: str,
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
) -> Dict:
    """
    Returns the complete review history grouped by card.
    Each key is `"{video_stem}:{card_index}"`.
    """
    if video_stem not in _progress:
        _progress[video_stem] = _load_progress(video_stem)

    data = _progress.get(video_stem, {})

    if session_id:
        data = {
            k: [r for r in v if r.get("session_id") == session_id]
            for k, v in data.items()
        }

    summary = {}
    for card_key, reviews in data.items():
        if not reviews:
            continue
        avg_conf = sum(r["confidence"] for r in reviews) / len(reviews)
        summary[card_key] = {
            "reviews":        reviews,
            "total_reviews":  len(reviews),
            "avg_confidence": round(avg_conf, 2),
            "correct_count":  sum(1 for r in reviews if r["correct"]),
            "needs_review":   avg_conf <= 3.0,
            "last_reviewed":  max(r["reviewed_at"] for r in reviews),
        }

    return {
        "video_stem":           video_stem,
        "session_id":           session_id,
        "cards":                summary,
        "total_cards_reviewed": len(summary),
    }


@app.get(
    "/progress/{video_stem}/due",
    summary="Cards due for review (spaced repetition queue)",
    tags=["Student Progress"],
)
async def get_due_cards(
    video_stem: str,
    session_id: Optional[str] = Query(None),
    limit:      int            = Query(20, ge=1, le=100),
) -> List[Dict]:
    """
    Returns flashcards that need review, ordered by priority:

    1. **Never reviewed** — appear first (most urgent)
    2. **Average confidence ≤ 3** — sorted oldest-reviewed first

    Each card includes its full question/answer plus review stats.
    Use this endpoint to drive a spaced-repetition study mode.
    """
    v          = _find_video(video_stem)
    flashcards = v.get("flashcards", []) if v else []

    if not flashcards:
        p = os.path.join(FLASHCARD_DIR, f"{video_stem}_flashcards.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                flashcards = json.load(f)

    if not flashcards:
        raise HTTPException(404, f"No flashcards found for '{video_stem}'.")

    if video_stem not in _progress:
        _progress[video_stem] = _load_progress(video_stem)
    prog = _progress.get(video_stem, {})

    due: List[Dict] = []

    for idx, card in enumerate(flashcards):
        card_key = f"{video_stem}:{idx}"
        reviews  = prog.get(card_key, [])
        if session_id:
            reviews = [r for r in reviews if r.get("session_id") == session_id]

        if not reviews:
            due.append({
                **card,
                "card_index":     idx,
                "card_key":       card_key,
                "total_reviews":  0,
                "avg_confidence": None,
                "last_reviewed":  None,
                "priority":       "never_reviewed",
            })
        else:
            avg_conf = sum(r["confidence"] for r in reviews) / len(reviews)
            if avg_conf <= 3.0:
                due.append({
                    **card,
                    "card_index":     idx,
                    "card_key":       card_key,
                    "total_reviews":  len(reviews),
                    "avg_confidence": round(avg_conf, 2),
                    "last_reviewed":  max(r["reviewed_at"] for r in reviews),
                    "priority":       "low_confidence",
                })

    # Sort: never_reviewed first, then by oldest last review
    due.sort(key=lambda c: (0 if c["priority"] == "never_reviewed" else 1,
                             c["last_reviewed"] or 0))
    return due[:limit]


# ──────────────────────────────────────────────────────────────────────────────
#  LANGUAGE INFO
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/languages", summary="Supported OCR + transcription languages", tags=["Info"])
async def list_languages() -> Dict:
    """
    Returns all languages supported by Whisper + EasyOCR in this system.
    Pass the ISO code to `?language=XX` on `POST /upload/image`.
    Video language is always auto-detected by Whisper.
    """
    langs = LanguageDetector.supported_languages()
    return {
        "total":     len(langs),
        "languages": langs,
        "note": (
            "For video uploads language is auto-detected from Whisper. "
            "For image uploads pass ?language=<code> to POST /upload/image."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  DIAGNOSTICS
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/diagnostics", summary="System health and endpoint reference", tags=["Health"])
async def diagnostics() -> JSONResponse:
    from video_pipeline.detection.ocr   import EASYOCR_AVAILABLE, TESSERACT_AVAILABLE
    from academic_system.slide_detector  import SKIMAGE_AVAILABLE
    from academic_system.deduplicator    import ST_AVAILABLE, SKLEARN_AVAILABLE
    from academic_system.knowledge_graph import NX_AVAILABLE

    return JSONResponse({
        "system":    "Multi-Modal Academic Intelligence System v3.0.0",
        "reasoning": config.reasoning_model_id,
        "backends": {
            "easyocr":               EASYOCR_AVAILABLE,
            "tesseract":             TESSERACT_AVAILABLE,
            "whisper":               WHISPER_AVAILABLE,
            "whisper_model":         config.whisper_model_size,
            "skimage_ssim":          SKIMAGE_AVAILABLE,
            "sentence_transformers": ST_AVAILABLE,
            "sklearn_tfidf":         SKLEARN_AVAILABLE,
            "networkx":              NX_AVAILABLE,
        },
        "deduplicator_backend": _deduplicator.backend,
        "slide_detection": {
            "hist_threshold":  config.slide_hist_threshold,
            "ssim_threshold":  config.slide_ssim_threshold,
            "min_gap_seconds": config.slide_min_seconds,
        },
        "outputs": [
            "① JSON         GET /results/video",
            "② Markdown     GET /results/notes/{stem}",
            "③ PDF          GET /results/pdf/{stem}",
            "④ Flashcards   GET /results/flashcards/{stem}",
            "⑤ MCQ Quiz     GET /results/quiz/{stem}",
            "⑥ Graph (D3)   GET /results/graph/{stem}  ← v3 NEW",
        ],
        "student_progress": [
            "PATCH /flashcards/{stem}/{card_index}/review",
            "GET   /progress/{stem}",
            "GET   /progress/{stem}/due",
        ],
        "all_endpoints": {
            "upload_video":  "POST   /upload/video",
            "upload_image":  "POST   /upload/image?language=en",
            "upload_audio":  "POST   /upload/audio",
            "status":        "GET    /status",
            "results_json":  "GET    /results/video",
            "results_image": "GET    /results/image",
            "results_audio": "GET    /results/audio",
            "study_notes":   "GET    /results/notes/{stem}",
            "pdf":           "GET    /results/pdf/{stem}",
            "flashcards":    "GET    /results/flashcards/{stem}",
            "quiz":          "GET    /results/quiz/{stem}",
            "graph":         "GET    /results/graph/{stem}",
            "frames":        "GET    /results/frames/{stem}",
            "latest":        "GET    /results/latest?n=10",
            "review":        "PATCH  /flashcards/{stem}/{card_index}/review",
            "progress":      "GET    /progress/{stem}",
            "due_cards":     "GET    /progress/{stem}/due",
            "languages":     "GET    /languages",
            "stop":          "POST   /stop",
            "clear":         "DELETE /results",
            "docs":          "GET    /docs",
        },
    })