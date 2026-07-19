"""
Multi-Modal Academic Intelligence System  v3.2.1 (main_v7-9.py)
================================================
v3.2.0 pipeline UNCHANGED + PostgreSQL/JWT Multi-User Auth layer from v3.6.5.

What's new in v3.2.1 (auth layer — zero pipeline changes)
----------------------------------------------------------
  ① PostgreSQL + SQLAlchemy DB — lectures, flashcards, quiz, student progress
     are persisted per-user across server restarts.
  ② JWT authentication — POST /auth/register  POST /auth/login
     All upload / result / generate endpoints require Bearer token.
  ③ Per-user result isolation — academic_results is now keyed by
     (user_id, video_path) so users never see each other's data.
  ④ save_lecture_to_db / save_flashcards_to_db called after Phase 2
     and after on-demand flashcard generation respectively.
  ⑤ DB fallback on GET endpoints — if the video is no longer in memory
     (e.g. after server restart) results are served from PostgreSQL.
  ⑥ Every pipeline function, constant, helper, and algorithm is
     identical to v3.2.0.  No logic was touched.

What's new in v3.2.0 (faster-whisper + chunked transcription)
--------------------------------------------------------------
  ① faster-whisper primary backend — CTranslate2 INT8 GPU inference.
  ② Segment-level streaming.
  ③ VAD filter.
  ④ Per-segment + total timeouts.
  ⑤ Double-timeout fix in main.
  ⑥ Diagnostics updated.
  ⑦ openai-whisper fallback.

What's new in v3.1.0 (1-hour video support + performance)
----------------------------------------------------------
  ① FPS cap for long videos — adaptive FPS scaling.
  ② Tiny Whisper by default.
  ③ Async OCR via run_in_executor.
  ④ Larger OCR batch (8, was 4).
  ⑤ Phase 2 smart sampling (max 40 frames to LLM).
  ⑥ Whisper streaming timeout.
  ⑦ Frame count hard cap.
  ⑧ Video duration detection at upload.

What's new in v3.0.3 (on-demand flashcards/quiz)
--------------------------------------------------
  ① Flashcards and quiz are NO LONGER generated during the upload pipeline.
  ② New endpoint:  POST /generate/flashcards/{video_stem}
  ③ New endpoint:  GET  /generate/flashcards/{video_stem}/status
  ④ GET /results/flashcards/{stem} and GET /results/quiz/{stem} unchanged.
  ⑤ prompt_cards_from_notes() added to prompts.py.

Outputs per video
  ① JSON API         GET /results/video
  ② Markdown notes   GET /results/notes/{stem}
  ③ PDF report       GET /results/pdf/{stem}
  ④ Q&A Flashcards   GET /results/flashcards/{stem}
  ⑤ MCQ Quiz         GET /results/quiz/{stem}
  ⑥ Knowledge Graph  GET /results/graph/{stem}

Auth endpoints
  POST /auth/register   { username, email, password }
  POST /auth/login      OAuth2 form  →  { access_token, token_type }

Run:
  uvicorn main_v7-9:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

# ── SSL fix — must be FIRST, before any network-using import ──────────────────
import ssl
import os

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE",      certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    _orig_create_default_ctx = ssl.create_default_context

    def _patched_ssl_context(*args, **kwargs):
        ctx = _orig_create_default_ctx(*args, **kwargs)
        ctx.load_verify_locations(certifi.where())
        return ctx

    ssl.create_default_context = _patched_ssl_context

except ImportError:
    import warnings
    warnings.warn(
        "certifi not installed. EasyOCR model downloads may fail with SSL errors. "
        "Run: pip install certifi",
        RuntimeWarning,
        stacklevel=1,
    )
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]

# ── Load environment variables from .env file ────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system environment variables

# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import json
import re
import shutil
import time
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import cv2
import numpy as np

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, Depends, status
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

# ── Internal modules ──────────────────────────────────────────────────────────
from video_pipeline.config1 import config
from video_pipeline.core.stream_manager import StreamManager
from video_pipeline.detection.ocr import OCRExtractor
from video_pipeline.reasoning.phi3_engine import Phi3Reasoner as LlamaReasoner
from video_pipeline.utils.device import setup_device
from video_pipeline.utils.logger import get_logger

from metadata.video_metadata import extract_video_metadata

from academic_system.prompts1 import (
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
    # ── v3.0.2: two-call split ────────────────────────────────────────────────
    prompt_metadata,
    prompt_study_notes_text, 
    prompt_cards_and_quiz,
    # ── v3.0.3: on-demand flashcard generation ────────────────────────────────
    prompt_cards_from_notes,
)
from academic_system.pdf_generator import generate_pdf_report
from academic_system.whisper_transcriber1 import (
    extract_audio, transcribe, convert_to_wav,
    release_models as release_whisper_models,   # v3.2.2: free VRAM before Phi-3
    WHISPER_AVAILABLE,
    FASTER_WHISPER_AVAILABLE,   # v3.2.0: faster-whisper backend flag
    OPENAI_WHISPER_AVAILABLE,   # v3.2.0: openai-whisper fallback flag
)
from academic_system.pdf_pipeline import (
    run_pdf_pipeline,
    init_pdf_pipeline_singletons,
    _check_pdf_backends,
)

# Keep strong references to background asyncio tasks so they aren't garbage collected
_background_tasks = set()

from live_lecture import router as live_router

# ── v3 additions ──────────────────────────────────────────────────────────────
from academic_system.slide_detector   import SlideChangeDetector
from academic_system.deduplicator     import SemanticDeduplicator
from academic_system.knowledge_graph  import KnowledgeGraphBuilder
from academic_system.language_support import LanguageDetector

# ── v3.2.1: Auth & Database ──────────────────────────────────────────────────
import auth
from auth import LocalUser as User
from minio_utils import minio_client

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  v3.1.0: LONG-VIDEO CONSTANTS  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

_PHASE2_MAX_FRAMES: int = int(os.environ.get("PHASE2_MAX_FRAMES", "10"))

_FPS_LONG_THRESHOLD_1: int = int(os.environ.get("FPS_LONG_THRESHOLD_1", str(20 * 60)))
_FPS_LONG_THRESHOLD_2: int = int(os.environ.get("FPS_LONG_THRESHOLD_2", str(40 * 60)))
_FPS_FOR_LONG_1: float     = float(os.environ.get("FPS_FOR_LONG_1", "0.2"))
_FPS_FOR_LONG_2: float     = float(os.environ.get("FPS_FOR_LONG_2", "0.1"))

_MAX_FRAMES_EXTRACT: int = int(os.environ.get("MAX_FRAMES_EXTRACT", "5"))

_WHISPER_TIMEOUT_SEC: int = int(os.environ.get("WHISPER_TIMEOUT_SEC", "600"))

_WHISPER_MODEL_SIZE: str = os.environ.get("WHISPER_MODEL_SIZE", "tiny")
_LOW_VRAM_SEQUENTIAL: bool = os.environ.get("LOW_VRAM_SEQUENTIAL", "1").strip().lower() not in {"0", "false", "no", "off"}


# ──────────────────────────────────────────────────────────────────────────────
#  APP  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Modal Academic Intelligence System",
    description=(
        "Transforms lecture videos, slide images, and audio recordings into "
        "structured student learning materials.\n\n"
        "**v3.2.1:** PostgreSQL + JWT auth layer — zero pipeline changes\n\n"
        "**v3.2.0:** faster-whisper CTranslate2 backend — 4× faster, INT8 GPU, "
        "VAD filter, segment streaming, per-chunk timeout; double-timeout fix\n\n"
        "**v3.1.0:** 1-hour video support — adaptive FPS, tiny Whisper, async OCR, "
        "Phase 2 frame cap, fast performance on RTX 3050\n\n"
        "**v3.0.3:** On-demand flashcard/quiz generation via "
        "POST /generate/flashcards/{stem}\n\n"
        "**v3.0.2:** 2-call Phase 2 split (fixes stalled flashcards/quiz/notes)\n\n"
        "**v3.0.1:** SSL/EasyOCR fix · OCR failure guard · phase-2 empty-data guard\n\n"
        "**v3.0.0:** slide change detection · semantic deduplication · "
        "knowledge graph · multi-language · student progress API"
    ),
    version="3.2.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(live_router)

@app.on_event("startup")
def on_startup():
    from database_v2 import init_db
    logger.info("Initializing database schema...")
    init_db()
    logger.info("Database initialized.")


# ──────────────────────────────────────────────────────────────────────────────
#  DIRECTORIES  (unchanged)
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
PROGRESS_DIR    = "student_progress"

SERVER_HOST      = os.environ.get("SERVER_HOST",      "127.0.0.1")
SERVER_PORT      = os.environ.get("SERVER_PORT",      "8000")
SERVER_BASE_PATH = os.environ.get("SERVER_BASE_PATH", "").rstrip("/")

for _d in (
    UPLOAD_DIR, AUDIO_DIR, IMAGE_DIR, FRAMES_BASE_DIR,
    NOTES_DIR, PDF_DIR, FLASHCARD_DIR, QUIZ_DIR,
    GRAPH_DIR, PROGRESS_DIR, config.output_dir,
):
    os.makedirs(_d, exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".txt"}


# ──────────────────────────────────────────────────────────────────────────────
#  GLOBAL STATE  (unchanged — academic_results keyed by video_path string)
# ──────────────────────────────────────────────────────────────────────────────

pipeline_task:    Optional[asyncio.Task] = None
pipeline_running: bool = False
pipeline_status:  str = "Idle"
pipeline_progress: int = 0  # 0 to 100

_pipeline_running_ref: List[bool] = [False]   # wraps the module-level pipeline_running

academic_results:      Dict[str, Dict[str, Any]] = {}
stream_frame_counters: Dict[str, int]            = {}

# v3 singletons
_deduplicator  = SemanticDeduplicator()
_graph_builder = KnowledgeGraphBuilder()
_lang_detector = LanguageDetector()

# Student progress store
_progress: Dict[str, Dict[str, List[Dict]]] = {}

# v3.0.3: per-stem flashcard generation tasks
_flashcard_tasks:  Dict[str, asyncio.Task]   = {}
_flashcard_states: Dict[str, Dict[str, Any]] = {}
_shared_llm: Optional[LlamaReasoner] = None
_shared_llm_ref: List[Optional[LlamaReasoner]] = [None]  # mutable ref for PDF pipeline to write back LLM

# v3.1.0: shared OCR executor
_ocr_executor: Optional[Any] = None


# ──────────────────────────────────────────────────────────────────────────────
#  v3.2.1: DB STARTUP
# ──────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    from database_v2 import init_db
    init_db()
    logger.info("[STARTUP] Database tables initialised.")
    
    # ── NEW: wire PDF pipeline to the same singletons ────────────────────────
    init_pdf_pipeline_singletons(_deduplicator, _graph_builder, _lang_detector)
    logger.info("[STARTUP] PDF pipeline singletons initialised.")


def get_or_load_shared_llm(context_label: str = "General") -> LlamaReasoner:
    """
    Checks for a shared LLM instance, synchronises it from the background ref
    if necessary, and loads it if it's missing.
    """
    global _shared_llm, _shared_llm_ref
    
    # 1. Sync from ref (e.g. if PDF pipeline loaded it in background)
    if _shared_llm is None and _shared_llm_ref[0] is not None:
        _shared_llm = _shared_llm_ref[0]
        logger.info(f"[{context_label}] Recovered _shared_llm from background ref.")

    # 2. Return if exists
    if _shared_llm is not None:
        logger.info(f"[{context_label}] Reusing shared Phi-3 LLM.")
        return _shared_llm

    # 3. Load fresh
    logger.info(f"[{context_label}] Loading Phi-3 into VRAM…")
    _shared_llm = LlamaReasoner(
        model_id       = config.reasoning_model_id,
        max_new_tokens = config.max_reasoning_tokens,
        device         = setup_device(),
        load_in_4bit   = config.phi3_load_in_4bit,
        adapter_path   = config.phi3_adapter_path or None,
    )
    # Ensure ref is also updated so other background tasks see it
    _shared_llm_ref[0] = _shared_llm
    return _shared_llm


# ──────────────────────────────────────────────────────────────────────────────
#  UTILITIES  (100 % unchanged from v3.2.0)
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
    # Use our custom redirector endpoint that pulls from MinIO
    filename = os.path.basename(rel)
    # Reconstruct the path part if needed (e.g. stem/filename)
    parent_dir = os.path.basename(os.path.dirname(rel))
    return f"{SERVER_BASE_PATH}/media/images/{parent_dir}/{filename}"


def _video_frames_dir(video_path: str) -> str:
    # Sanitize and shorten stem to avoid MAX_PATH issues on Windows
    full_stem = Path(video_path).stem
    # Keep UUID prefix if present, then shorten title, remove special chars
    parts = full_stem.split('_', 1)
    if len(parts) > 1:
        prefix, title = parts[0], parts[1]
        clean_title = "".join(c if c.isalnum() else "_" for c in title)[:50]
        short_stem = f"{prefix}_{clean_title}"
    else:
        short_stem = "".join(c if c.isalnum() else "_" for c in full_stem)[:100]
        
    d = os.path.join(FRAMES_BASE_DIR, short_stem)
    os.makedirs(d, exist_ok=True)
    return d



def save_frame(
    frame: np.ndarray,
    frame_id: int,
    timestamp: float,
    video_path: str,
) -> Tuple[str, str]:
    frames_dir = _video_frames_dir(video_path)
    filename   = f"frame_{frame_id:05d}_t{timestamp:.3f}s.jpg"
    path       = os.path.join(frames_dir, filename)
    success    = cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        logger.error(f"[Save] cv2.imwrite failed for {path} (Path too long or invalid chars?)")
    rel = os.path.relpath(path)

    
    # Upload to MinIO
    try:
        from minio_utils import minio_client
        stem = Path(video_path).stem
        object_name = f"images/{stem}/{filename}"
        minio_client.upload_file(path, object_name)
    except Exception as e:
        logger.error(f"Failed to upload frame {filename} to MinIO: {e}")

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


def _render_progress_bar(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    ratio = max(0.0, min(1.0, current / total))
    filled = int(ratio * width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _log_pipeline_progress(prefix: str, current: int, total: int, extra: str = "") -> None:
    bar = _render_progress_bar(current, total)
    pct = 0.0 if total <= 0 else (current / total) * 100.0
    msg = f"{prefix} {bar} {pct:5.1f}% ({current}/{total})"
    if extra:
        msg = f"{msg} | {extra}"
    logger.info(msg)


def _vram_snapshot() -> str:
    try:
        if not torch.cuda.is_available():
            return "cuda=unavailable"
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        allocated = torch.cuda.memory_allocated(idx) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(idx) / (1024 ** 3)
        total = props.total_memory / (1024 ** 3)
        free_est = max(total - reserved, 0.0)
        return (
            f"cuda:{idx} alloc={allocated:.2f}GB "
            f"reserved={reserved:.2f}GB free~={free_est:.2f}GB total={total:.2f}GB"
        )
    except Exception as exc:
        return f"vram=unknown ({exc})"


def _log_vram(event: str) -> None:
    logger.info(f"[VRAM] {event} | {_vram_snapshot()}")


def stem_path(video_path: str) -> str:
    return Path(video_path).stem


def _get_video_duration_sec(video_path: str) -> Optional[float]:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps_cv   = cap.get(cv2.CAP_PROP_FPS)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps_cv > 0 and n_frames > 0:
            return n_frames / fps_cv
        return None
    except Exception:
        return None


def _adaptive_fps(duration_sec: Optional[float]) -> float:
    if duration_sec is None:
        return config.fps
    if duration_sec > _FPS_LONG_THRESHOLD_2:
        return _FPS_FOR_LONG_2
    if duration_sec > _FPS_LONG_THRESHOLD_1:
        return _FPS_FOR_LONG_1
    return config.fps


_FIXED_VIDEO_SAMPLE_POINTS: Tuple[float, ...] = (0.10, 0.25, 0.60, 0.85, 0.95)


def _get_fixed_sample_timestamps(duration_sec: Optional[float]) -> List[float]:
    if not duration_sec or duration_sec <= 0:
        return []
    return [round(duration_sec * pct, 3) for pct in _FIXED_VIDEO_SAMPLE_POINTS]


def _extract_fixed_frames(video_path: str, timestamps: List[float]) -> List[Tuple[np.ndarray, float]]:
    frames: List[Tuple[np.ndarray, float]] = []
    if not timestamps:
        return frames

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames

    try:
        fps_cv = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for timestamp in timestamps:
            if fps_cv > 0 and total_frames > 0:
                frame_index = min(max(int(round(timestamp * fps_cv)), 0), total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(timestamp, 0.0) * 1000.0)

            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning(f"[Fixed Sampling] Failed to read frame for {video_path} at {timestamp:.3f}s")
                continue
            frames.append((frame, timestamp))
    finally:
        cap.release()

    return frames


# ── LLM helpers (unchanged) ───────────────────────────────────────────────────

_PHI3_CTX   = int(os.environ.get("PHI3_CONTEXT_LENGTH", "4096"))
_CTX_MARGIN = 64

_MAX_TOKENS_META  = int(os.environ.get("PHI3_MAX_TOKENS_META",  "300"))
_MAX_TOKENS_NOTES = int(os.environ.get("PHI3_MAX_TOKENS_NOTES", "700"))
_MAX_TOKENS_CARDS = int(os.environ.get("PHI3_MAX_TOKENS_CARDS", "2048"))


def _safe_max_tokens(llm: LlamaReasoner, prompt: str, desired: int) -> int:
    ctx = int(
        getattr(llm, "context_length", None)
        or getattr(llm, "max_position_embeddings", None)
        or config.phi3_context_length
    )
    prompt_tokens_estimate = max(1, len(prompt) // 4)
    available = ctx - prompt_tokens_estimate - _CTX_MARGIN
    safe = max(64, min(desired, available))
    if safe < desired:
        logger.warning(
            f"[main] Token budget capped: desired={desired} "
            f"prompt_est={prompt_tokens_estimate} ctx={ctx} -> safe={safe}"
        )
    return safe


def _llm_reason(llm: LlamaReasoner, prompt: str, max_tokens: int) -> Any:
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
    safe     = _safe_max_tokens(llm, prompt, max_tokens)
    original = getattr(llm, "max_new_tokens", None)
    try:
        if original is not None:
            llm.max_new_tokens = safe
        return llm.reason_text(prompt)
    finally:
        if original is not None:
            llm.max_new_tokens = original


#
# NOTE: JSON helpers live further down in the file (near the on-demand
# flashcard generation code). Keep a single implementation to avoid
# accidental behavior differences from duplicate defs.
#


def _extract_first_json_object(raw: str) -> Dict:
    if not raw:
        return {}

    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned.strip())

    start = cleaned.find("{")
    if start == -1:
        return {}

    depth  = 0
    in_str = False
    escape = False

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
                candidate = cleaned[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    try:
                        obj = json.loads(cleaned)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                return {}
    return {}


def _partial_json_list(raw: str) -> List[Dict]:
    if not raw or not raw.strip():
        return []

    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

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

    objects: List[Dict] = []
    depth  = 0
    in_str = False
    escape = False
    start  = None

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


def _extract_json_array_by_key(raw: str, key: str) -> Optional[str]:
    """
    Extract the JSON array text that follows a top-level key, e.g.:
      "flashcards": [ ... ]
    Works even if the full document is malformed elsewhere (common with LLM output).
    """
    if not raw:
        return None
    # Find the key (quoted) and then the first '[' after the colon.
    m = re.search(rf'"{re.escape(key)}"\s*:', raw)
    if not m:
        return None
    i = m.end()
    bracket_start = raw.find("[", i)
    if bracket_start == -1:
        return None

    depth = 0
    in_str = False
    escape = False
    start = bracket_start
    for j in range(bracket_start, len(raw)):
        ch = raw[j]
        if escape:
            escape = False
            continue
        if in_str and ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return raw[start : j + 1]
    return None


def _recover_list_of_objects(array_text: str) -> List[Dict]:
    """
    Best-effort recovery for an array of JSON objects when the array can't be
    parsed as a whole. Extracts each {...} and json.loads it individually.
    """
    if not array_text:
        return []

    items: List[Dict] = []
    cleaned = array_text.strip()
    # Ensure we're scanning inside a list.
    if "[" in cleaned:
        cleaned = cleaned[cleaned.find("[") + 1 :]
    if "]" in cleaned:
        cleaned = cleaned[: cleaned.rfind("]")]

    depth = 0
    in_str = False
    escape = False
    obj_start: Optional[int] = None

    for idx, ch in enumerate(cleaned):
        if escape:
            escape = False
            continue
        if in_str and ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                obj_start = idx
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                candidate = cleaned[obj_start : idx + 1].strip()
                candidate = _repair_json(candidate)
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        items.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = None

    return items


def _salvage_flashcards_quiz_from_malformed(raw_text: str) -> Dict[str, Any]:
    """
    Salvage flashcards/quiz lists from malformed "almost JSON" outputs.
    Returns a dict with keys: flashcards, quiz (always present as lists).
    """
    cleaned = _robust_strip_fences(raw_text or "")
    cleaned = _repair_json(cleaned)

    flashcards: List[Dict] = []
    quiz: List[Dict] = []

    fc_array = _extract_json_array_by_key(cleaned, "flashcards")
    if fc_array:
        try:
            parsed = json.loads(_repair_json(fc_array))
            if isinstance(parsed, list):
                flashcards = [x for x in parsed if isinstance(x, dict)]
        except json.JSONDecodeError:
            flashcards = _recover_list_of_objects(fc_array)

    quiz_array = _extract_json_array_by_key(cleaned, "quiz")
    if quiz_array:
        try:
            parsed = json.loads(_repair_json(quiz_array))
            if isinstance(parsed, list):
                quiz = [x for x in parsed if isinstance(x, dict)]
        except json.JSONDecodeError:
            quiz = _recover_list_of_objects(quiz_array)

    return {"flashcards": flashcards, "quiz": quiz}


def _find_any_lecture(stem: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    for v in academic_results.values():
        if v.get("input_type") not in ("video", "images", "pdf", "document", "live"):
            continue

        candidate_paths = [
            v.get("video_path", ""),
            v.get("pdf_path", ""),
        ]
        if any(Path(p).stem == stem for p in candidate_paths if p) and (
            user_id is None or v.get("user_id") == user_id
        ):
            return v
    return None


def _sample_frames_for_phase2(frames: List[Dict], max_n: int = _PHASE2_MAX_FRAMES) -> List[Dict]:
    if len(frames) <= max_n:
        return frames
    step = len(frames) / max_n
    return [frames[int(i * step)] for i in range(max_n)]


# ── Student progress helpers (unchanged) ──────────────────────────────────────

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


def _stem_input_type(stem: str) -> str:
    if stem.endswith("_img"):
        return "images"
    if stem.endswith("_doc"):
        return "document"
    return "video"


def _display_name_from_stem(stem: str) -> str:
    display = stem.split("_", 1)[-1] if "_" in stem else stem
    for suffix in ("_img", "_doc"):
        if display.endswith(suffix):
            display = display[: -len(suffix)]
    return display or stem


def _artifact_path(directory: str, stem: str, suffix: str) -> str:
    return os.path.join(directory, f"{stem}{suffix}")


def _find_uploaded_source(stem: str, input_type: str) -> Optional[str]:
    source_dir = UPLOAD_DIR if input_type in {"video", "document"} else IMAGE_DIR
    allowed = VIDEO_EXTENSIONS if input_type == "video" else DOCUMENT_EXTENSIONS

    if input_type == "images":
        for ext in IMAGE_EXTENSIONS:
            candidate = os.path.join(IMAGE_DIR, f"{stem}{ext}")
            if os.path.isfile(candidate):
                return candidate
        return None

    for name in os.listdir(source_dir):
        path = os.path.join(source_dir, name)
        if not os.path.isfile(path):
            continue
        if Path(path).suffix.lower() not in allowed:
            continue
        if Path(path).stem == stem:
            return path
    return None


def _load_json_if_exists(path: str) -> Any:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _collect_disk_lecture_stems() -> set[str]:
    stems: set[str] = set()
    suffix_map = [
        (NOTES_DIR, "_study_notes.md"),
        (PDF_DIR, "_academic_report.pdf"),
        (FLASHCARD_DIR, "_flashcards.json"),
        (QUIZ_DIR, "_quiz.json"),
        (GRAPH_DIR, "_knowledge_graph.json"),
    ]
    for directory, suffix in suffix_map:
        for name in os.listdir(directory):
            if name.endswith(suffix):
                stems.add(name[: -len(suffix)])
    for name in os.listdir(FRAMES_BASE_DIR):
        path = os.path.join(FRAMES_BASE_DIR, name)
        if os.path.isdir(path):
            stems.add(name)
    return stems


def _disk_lecture_state(stem: str) -> Dict[str, Any]:
    input_type = _stem_input_type(stem)
    notes_path = _artifact_path(NOTES_DIR, stem, "_study_notes.md")
    pdf_path = _artifact_path(PDF_DIR, stem, "_academic_report.pdf")
    flashcards_path = _artifact_path(FLASHCARD_DIR, stem, "_flashcards.json")
    quiz_path = _artifact_path(QUIZ_DIR, stem, "_quiz.json")
    graph_path = _artifact_path(GRAPH_DIR, stem, "_knowledge_graph.json")
    frames_index_path = os.path.join(FRAMES_BASE_DIR, stem, "frames_index.json")

    flashcards = _load_json_if_exists(flashcards_path) or []
    quiz = _load_json_if_exists(quiz_path) or []
    frames_index = _load_json_if_exists(frames_index_path) or []

    mtimes = [
        os.path.getmtime(path)
        for path in (
            notes_path,
            pdf_path,
            flashcards_path,
            quiz_path,
            graph_path,
            frames_index_path,
        )
        if os.path.exists(path)
    ]

    return {
        "source": "disk",
        "input_type": input_type,
        "audio_ready": input_type == "video",
        "summary_ready": os.path.isfile(notes_path) or os.path.isfile(pdf_path),
        "study_notes_ready": os.path.isfile(notes_path),
        "pdf_ready": os.path.isfile(pdf_path),
        "flashcards_generation_state": "done" if flashcards or quiz else "idle",
        "flashcards_ready": bool(flashcards),
        "flashcard_count": len(flashcards),
        "quiz_ready": bool(quiz),
        "quiz_count": len(quiz),
        "graph_ready": os.path.isfile(graph_path),
        "video_stem": stem,
        "display_name": _display_name_from_stem(stem),
        "lecture_title": _display_name_from_stem(stem),
        "subject_area": None,
        "difficulty": None,
        "pipeline_error": None,
        "created_at": max(mtimes) if mtimes else 0,
        "frames_collected": len(frames_index),
        "frames_index": frames_index if input_type == "images" else None,
        "flashcard_generate_url": f"POST /generate/flashcards/{stem}",
    }


# ──────────────────────────────────────────────────────────────────────────────
#  IMAGE ANALYSIS  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def process_image_academic(
    image_path: str,
    ocr:        OCRExtractor,
    llm:        LlamaReasoner,
    lang_info:  Optional[Dict] = None,
    frame_id:   int = 1,
    timestamp:  float = 0.0,
) -> Dict[str, Any]:
    """
    Full per-image analysis: OCR → LLM explanation → structured academic content.
    Returns the same schema as per_frame_details for pipeline compatibility.
    """
    from academic_system.prompts1 import prompt_image_explain

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Cannot read image file: {image_path}")

    filename = Path(image_path).name

    # ── OCR ──────────────────────────────────────────────────────────────────
    raw_ocr  = ocr.extract(frame, config.ocr_confidence_threshold)
    ocr_text = ocr_to_text(raw_ocr)

    # ── LLM: rich per-image explanation ─────────────────────────────────────
    img_prompt = prompt_image_explain(ocr_text, frame_id, filename)
    if lang_info:
        img_prompt = _lang_detector.patch_prompt(img_prompt, lang_info)

    academic_content = serialize(_llm_reason(llm, img_prompt, 500)) or {}

    # Fallback if LLM returns nothing
    if not academic_content:
        academic_content = {
            "image_title":    filename,
            "content_type":   "unknown",
            "importance":     "medium",
            "description":    ocr_text[:300] if ocr_text else "No content detected.",
            "key_concepts":   [],
            "formulas":       [],
            "bullet_points":  [],
            "content_summary": ocr_text[:200] if ocr_text else "",
        }

    logger.info(
        f"[Image {frame_id}] '{academic_content.get('image_title', filename)}' — "
        f"type={academic_content.get('content_type','?')} "
        f"importance={academic_content.get('importance','?')}"
    )

    return {
        "frame_id":         frame_id,
        "timestamp":        timestamp,
        "filename":         filename,
        "frame_path":       os.path.relpath(image_path),
        "frame_url":        _make_frame_url(os.path.relpath(image_path)),
        "academic_content": academic_content,
        "ocr_text":         ocr_text,
        "ocr_raw":          serialize(raw_ocr),
        # Convenience fields surfaced directly for easy API access
        "image_title":      academic_content.get("image_title", filename),
        "description":      academic_content.get("description", ""),
        "content_type":     academic_content.get("content_type", "unknown"),
        "key_concepts":     academic_content.get("key_concepts", []),
        "bullet_points":    academic_content.get("bullet_points", []),
        "formulas":         academic_content.get("formulas", []),
        "content_summary":  academic_content.get("content_summary", ""),
    }


async def run_image_batch_pipeline(
    images:     List[str],
    user_id:    int,
    batch_stem: str,
) -> None:
    """
    Full image-batch pipeline — mirrors the video pipeline quality:
      Phase 1 — OCR + per-image LLM explanation for every uploaded image
      Phase 2 — Batch metadata (Call 1) + Study notes (Call 2)
      Phase 3 — Knowledge graph + PDF report
      Flashcard/quiz available on-demand via POST /generate/flashcards/{stem}
    """
    from academic_system.prompts1 import (
        prompt_image_batch_metadata,
        prompt_image_study_notes,
    )

    global pipeline_running, _pipeline_running_ref, _shared_llm
    pipeline_running = _pipeline_running_ref[0] = True
    logger.info(
        f"🚀 IMAGE PIPELINE STARTED — stem={batch_stem} "
        f"files={len(images)}"
    )

    # ── initialise result record immediately so /status works ─────────────────
    vr: Dict[str, Any] = {
        "user_id":               user_id,
        "input_type":            "images",
        "images":                images,
        "video_path":            batch_stem,   # kept for _find_any_lecture compat
        "frames_index":          [],
        "per_frame_details":     [],
        "lecture_summary":       {},
        "audio_topics":          {},
        "study_notes":           None,
        "flashcards":            [],
        "quiz":                  [],
        "knowledge_graph":       None,
        "pdf_report_path":       None,
        "deduped_concepts":      [],
        "deduped_formulas":      [],
        "error":                 None,
    }
    academic_results[batch_stem] = vr

    try:
        # ── Models ────────────────────────────────────────────────────────────
        logger.info("[Image Pipeline] Initialising OCR engine (CPU)…")
        ocr = OCRExtractor(use_gpu=False)   # always CPU — saves VRAM for Phi-3

        llm = get_or_load_shared_llm("Image Pipeline")

        lang_info = _lang_detector.from_code("en")

        # ── PHASE 1: per-image OCR + LLM explanation ─────────────────────────
        logger.info(f"[Image Pipeline] Phase 1 — analysing {len(images)} image(s)…")
        image_analyses: List[Dict] = []

        for i, image_path in enumerate(images):
            logger.info(
                f"[Image {i+1}/{len(images)}] "
                f"Analysing: {Path(image_path).name}"
            )
            try:
                result = process_image_academic(
                    image_path = image_path,
                    ocr        = ocr,
                    llm        = llm,
                    lang_info  = lang_info,
                    frame_id   = i + 1,
                    timestamp  = float(i * 5),
                )
                image_analyses.append(result)
                vr["per_frame_details"].append(result)
                logger.info(
                    f"[Image {i+1}/{len(images)}] ✅ "
                    f"'{result.get('image_title', Path(image_path).name)}' "
                    f"— {result.get('content_type','?')}"
                )
            except Exception as exc:
                logger.error(
                    f"[Image {i+1}/{len(images)}] ❌ Failed: {exc}",
                    exc_info=True,
                )
                # Insert a placeholder so indexing stays consistent
                image_analyses.append({
                    "frame_id":         i + 1,
                    "filename":         Path(image_path).name,
                    "frame_path":       image_path,
                    "academic_content": {"importance": "low"},
                    "ocr_text":         "",
                    "error":            str(exc),
                })

        vr["frames_index"] = image_analyses

        if not image_analyses:
            raise RuntimeError("No images could be processed.")

        # ── PHASE 2, Call 1: batch metadata ───────────────────────────────────
        logger.info("[Image Pipeline] Phase 2, Call 1 — extracting batch metadata…")
        meta: Dict[str, Any] = {}
        try:
            call1_prompt = _lang_detector.patch_prompt(
                prompt_image_batch_metadata(
                    batch_stem      = batch_stem,
                    image_analyses  = image_analyses,
                    sample_n        = min(5, len(image_analyses)),
                    max_concepts    = 8,
                ),
                lang_info,
            )
            logger.info(
                f"[Image Pipeline] Call 1 prompt: {len(call1_prompt)} chars "
                f"(~{len(call1_prompt)//4} tokens)"
            )
            meta = serialize(_llm_reason(llm, call1_prompt, _MAX_TOKENS_META)) or {}

            if not meta:
                raw_text = getattr(llm, "_last_raw_output", "") or ""
                if raw_text:
                    meta = _extract_first_json_object(raw_text) or {}
                    if meta:
                        logger.info(f"[Image Pipeline] Call 1 JSON recovery: {list(meta.keys())}")

        except Exception as exc:
            logger.error(f"[Image Pipeline] Call 1 (metadata) failed: {exc}", exc_info=True)

        lecture_title = meta.get("lecture_title", batch_stem)
        subject_area  = meta.get("subject_area", "General")
        difficulty    = meta.get("difficulty", "")
        topics        = meta.get("topics", [])
        key_concepts  = meta.get("key_concepts", [])
        outcomes      = meta.get("learning_outcomes", [])
        summary       = meta.get("summary", "")

        lecture_summary = {
            "lecture_title":     lecture_title,
            "subject_area":      subject_area,
            "main_topics":       topics,
            "learning_outcomes": outcomes,
            "summary":           summary,
            "difficulty_level":  difficulty,
        }
        audio_topics = {
            "lecture_title":    lecture_title,
            "subject_area":     subject_area,
            "topics_covered":   topics,
            "key_concepts":     [{"concept": c, "explanation": ""} for c in key_concepts],
            "important_points": outcomes,
            "summary":          summary,
        }
        logger.info(
            f"[Image Pipeline] Call 1 done — "
            f"title='{lecture_title[:60]}' subject='{subject_area}'"
        )

        # ── PHASE 2, Call 2: study notes ──────────────────────────────────────
        logger.info("[Image Pipeline] Phase 2, Call 2 — generating study notes…")
        notes_md = ""
        try:
            call2_prompt = _lang_detector.patch_prompt(
                prompt_image_study_notes(
                    lecture_title     = lecture_title,
                    subject_area      = subject_area,
                    difficulty        = difficulty,
                    topics            = topics,
                    key_concepts      = key_concepts,
                    learning_outcomes = outcomes,
                    summary           = summary,
                    image_analyses    = image_analyses,
                ),
                lang_info,
            )
            logger.info(
                f"[Image Pipeline] Call 2 prompt: {len(call2_prompt)} chars "
                f"(~{len(call2_prompt)//4} tokens)"
            )
            notes_md = _llm_reason_text(llm, call2_prompt, _MAX_TOKENS_NOTES)
            logger.info(f"[Image Pipeline] Call 2 done — {len(notes_md)} chars generated.")

        except Exception as exc:
            logger.error(f"[Image Pipeline] Call 2 (notes) failed: {exc}", exc_info=True)

        # Fallback notes if LLM returned nothing
        if not notes_md or not notes_md.strip():
            lines = [f"# Study Notes: {lecture_title}", ""]
            if summary:
                lines += ["## Overview", "", summary, ""]
            if topics:
                lines += ["## Topics", ""] + [f"- {t}" for t in topics] + [""]
            if key_concepts:
                lines += ["## Key Concepts", ""] + [f"- {c}" for c in key_concepts] + [""]
            # Add per-image section
            lines += ["## Image Analyses", ""]
            for img in image_analyses:
                ac = img.get("academic_content", {})
                if ac.get("importance") == "low":
                    continue
                lines.append(f"### {img.get('image_title', img.get('filename', 'Image'))}")
                if img.get("description"):
                    lines.append(img["description"])
                if img.get("bullet_points"):
                    lines += [f"- {bp}" for bp in img["bullet_points"]]
                lines.append("")
            notes_md = "\n".join(lines)
            logger.info("[Image Pipeline] Using fallback notes from metadata.")
        elif not notes_md.lstrip().startswith("#"):
            notes_md = f"# Study Notes: {lecture_title}\n\n" + notes_md

        notes_path = write_text_file(
            os.path.join(NOTES_DIR, f"{batch_stem}_study_notes.md"),
            notes_md,
        )

        # ── Deduplication ─────────────────────────────────────────────────────
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=2) as _pool:
            f_concepts = _pool.submit(_deduplicator.deduplicate_concepts, image_analyses)
            f_formulas = _pool.submit(_deduplicator.deduplicate_formulas, image_analyses)
            deduped_concepts = f_concepts.result()
            deduped_formulas = f_formulas.result()

        logger.info(
            f"[Image Pipeline] Dedup: "
            f"{len(deduped_concepts)} concepts, {len(deduped_formulas)} formulas."
        )

        # ── PHASE 3: Knowledge graph + PDF ────────────────────────────────────
        logger.info("[Image Pipeline] Phase 3 — building knowledge graph and PDF…")

        graph_d3   = None
        graph_path = None
        pdf_path   = None

        try:
            graph      = _graph_builder.build(image_analyses, audio_topics, lecture_summary)
            graph_d3   = _graph_builder.to_d3_json(graph)
            graph_path = _graph_builder.save(
                graph,
                os.path.join(GRAPH_DIR, f"{batch_stem}_knowledge_graph.json"),
            )
            logger.info("[Image Pipeline] Knowledge graph built.")
        except Exception as exc:
            logger.error(f"[Image Pipeline] Knowledge graph failed: {exc}", exc_info=True)

        try:
            pdf_path = generate_pdf_report(
                video_path      = batch_stem,
                pdf_dir         = PDF_DIR,
                lecture_summary = lecture_summary,
                audio_topics    = audio_topics,
                frame_analyses  = image_analyses,
                flashcards      = [],
                transcript_text = "",
            )
            logger.info(f"[Image Pipeline] PDF report: {pdf_path}")
        except Exception as exc:
            logger.error(f"[Image Pipeline] PDF failed: {exc}", exc_info=True)

        # ── Finalize result record ─────────────────────────────────────────────
        vr.update({
            "lecture_summary":       lecture_summary,
            "audio_topics":          audio_topics,
            "study_notes":           notes_md,
            "study_notes_path":      notes_path,
            "knowledge_graph":       graph_d3,
            "knowledge_graph_path":  graph_path,
            "pdf_report_path":       pdf_path,
            "deduped_concepts":      deduped_concepts,
            "deduped_formulas":      deduped_formulas,
            "total_frames_analysed": len(image_analyses),
        })

        # ── Register flashcard state so on-demand endpoint works ──────────────
        _flashcard_states[batch_stem] = {
            "state":           "idle",
            "flashcard_count": 0,
            "quiz_count":      0,
            "error":           None,
        }

        # ── v3.2.1: persist image batch to Video DB ─────────────────────────
        if user_id:
            from db_actions import save_pipeline_result_to_db
            # Use the first image as 'video_path' for DB record, it just needs a representative path
            rep_path = images[0] if images else batch_stem
            save_pipeline_result_to_db(user_id, rep_path, vr, batch_stem)
            logger.info(f"[Image Pipeline] Persisted '{batch_stem}' to DB for user {user_id}.")

        logger.info(f"[Image Pipeline] Completed processing for '{batch_stem}'.")

        logger.info(
            f"✅ IMAGE PIPELINE COMPLETE — {batch_stem}\n"
            f"   {len(image_analyses)} images analysed\n"
            f"   Notes: {notes_path}\n"
            f"   PDF:   {pdf_path}\n"
            f"   Graph: {graph_path}\n"
            f"   Flashcards: POST /generate/flashcards/{batch_stem}"
        )

    except Exception as exc:
        logger.error(
            f"❌ IMAGE PIPELINE FAILED — {batch_stem}: {exc}",
            exc_info=True,
        )
        vr["error"] = str(exc)

    finally:
        pipeline_running = _pipeline_running_ref[0] = False


# ──────────────────────────────────────────────────────────────────────────────
#  ON-DEMAND FLASHCARD GENERATION  (v3.0.3, unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def _repair_json(s: str) -> str:
    """
    Repair common LLM JSON syntax errors:
      - Typos in property names (e.g., "questionimoine" -> "question")
      - Missing commas between properties (aggressive pass)
      - Incomplete or malformed structure
    """
    if not s:
        return s
    
    # Fix common property name typos first
    repairs = [
        (r'"questionimoine"\s*:', '"question":'),
        (r'"questin"\s*:', '"question":'),
        (r'"questoin"\s*:', '"question":'),
        (r'"answr"\s*:', '"answer":'),
        (r'"answeer"\s*:', '"answer":'),
        (r'"optons"\s*:', '"options":'),
        (r'"correctanswer"\s*:', '"correct_answer":'),
        (r'"correct_answeer"\s*:', '"correct_answer":'),
        (r'"explnation"\s*:', '"explanation":'),
        (r'"explantion"\s*:', '"explanation":'),
    ]
    
    for pattern, replacement in repairs:
        s = re.sub(pattern, replacement, s, flags=re.IGNORECASE)
    
    # Aggressive comma insertion: any line ending with " followed by line starting with "
    # This catches: "value"\n"key": and all similar patterns with missing commas
    s = re.sub(r'(")\s*\n(\s*")', r'\1,\n\2', s)

    # Remove trailing commas before closing braces/brackets
    s = re.sub(r",\s*([\]}])", r"\1", s)
     
    return s


def _robust_strip_fences(raw: str) -> str:
    """
    Strip markdown code fences from LLM output.
    Handles all of:
        ```json\\n{...}\\n```
        ```\\n{...}\\n```
        {... (no fences at all)
    """
    if not raw:
        return raw
    s = raw.strip()
    # Remove opening fence (```json or ```) on its own line or at very start
    s = re.sub(r"^```(?:json)?\s*\n?", "", s, flags=re.IGNORECASE)
    # Remove closing fence at end
    s = re.sub(r"\n?```\s*$", "", s, flags=re.IGNORECASE)
    return s.strip()


async def _run_flashcard_generation_with_llm(video_stem: str, user_id: Optional[str] = None) -> None:
    """Reuses the pipeline's already-loaded LLM if available."""
    global _shared_llm, _shared_llm_ref
    state = _flashcard_states[video_stem]

    try:
        llm = get_or_load_shared_llm(f"Flashcards/{video_stem}")
    except Exception as exc:
        logger.error(f"[Flashcards/{video_stem}] LLM load FAILED: {exc}", exc_info=True)
        state["state"] = "failed"
        state["error"] = f"LLM load failed: {exc}"
        return

    await _run_flashcard_generation(video_stem, llm, user_id=user_id)


async def _run_flashcard_generation(
    video_stem: str,
    llm: LlamaReasoner,
    user_id: Optional[str] = None,         # v3.2.1: used for DB persist
) -> None:
    """
    Fixed version of _run_flashcard_generation.
    Fixes:
      BUG 1 — Robust multi-pass JSON fence stripping before json.loads
      BUG 2 — ALWAYS persist to DB, even if lists are empty
    """
    state = _flashcard_states[video_stem]
    state["state"] = "running"
    logger.info(f"[Flashcards/{video_stem}] Generation started.")

    flashcards: List[Dict] = []
    quiz:       List[Dict] = []

    try:
        vr = _find_any_lecture(video_stem, user_id)
        if vr is None:
            raise ValueError(
                f"No pipeline result found for stem '{video_stem}'. "
                "Upload the video first."
            )

        notes_md: str = vr.get("study_notes", "") or ""
        if not notes_md:
            notes_path = os.path.join(NOTES_DIR, f"{video_stem}_study_notes.md")
            if os.path.isfile(notes_path):
                with open(notes_path, encoding="utf-8") as f:
                    notes_md = f.read()

        if not notes_md.strip():
            raise ValueError(
                "Study notes are not yet generated. "
                "Wait for the upload pipeline to complete first."
            )

        lecture_summary: Dict  = vr.get("lecture_summary", {})
        deduped_concepts: List[str] = vr.get("deduped_concepts", [])
        deduped_formulas: List[str] = vr.get("deduped_formulas", [])

        transcript: str = ""
        audio_analysis = vr.get("audio_analysis") or {}
        if isinstance(audio_analysis, dict):
            transcript = audio_analysis.get("text", "")

        prompt = prompt_cards_from_notes(
            notes_md          = notes_md,
            lecture_title     = lecture_summary.get("lecture_title", video_stem),
            subject_area      = lecture_summary.get("subject_area", "General"),
            key_concepts      = deduped_concepts,
            formulas          = deduped_formulas,
            transcript        = transcript,
            topics            = lecture_summary.get("main_topics", []),
            learning_outcomes = lecture_summary.get("learning_outcomes", []),
        )

        logger.info(
            f"[Flashcards/{video_stem}] Prompt: {len(prompt)} chars "
            f"(~{len(prompt)//4} tokens)"
        )

        raw_text: str = _llm_reason_text(llm, prompt, _MAX_TOKENS_CARDS) or ""
        llm._last_raw_output = raw_text

        # ── FIX: robust multi-pass JSON extraction ────────────────────────────
        raw: Dict = {}
 
        if raw_text:
            # Pass 1: strip fences robustly (handles leading whitespace) + repair
            cleaned  = _robust_strip_fences(raw_text)
            cleaned  = _repair_json(cleaned)
 
            # Pass 1: direct json.loads on the cleaned+repaired string
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    raw = parsed
                    logger.info(f"[Flashcards/{video_stem}] JSON parsed cleanly (pass 1).")
            except json.JSONDecodeError:
                pass
 
            # Pass 2: scan for first {{...}} object
            if not raw:
                logger.warning(
                    f"[Flashcards/{video_stem}] Pass 1 (direct parse) failed. "
                    f"Raw ({len(raw_text)} chars): {raw_text[:200]!r}"
                )
                raw = _extract_first_json_object(cleaned) or {}
                if raw:
                    logger.info(
                        f"[Flashcards/{video_stem}] JSON recovered via object scan (pass 2): "
                        f"keys={list(raw.keys())}"
                    )
 
            # Pass 3: partial list recovery (handles truncated output)
            if not raw:
                partial = _partial_json_list(cleaned)
                if partial:
                    logger.info(
                        f"[Flashcards/{video_stem}] Partial recovery: "
                        f"{len(partial)} item(s) from truncated output (pass 3)."
                    )
                    raw = {"flashcards": partial, "quiz": []}
 
            if not raw:
                logger.error(
                    f"[Flashcards/{video_stem}] All parse passes failed. "
                    f"Full raw output:\n{raw_text}"
                )

        # Pass 4: salvage arrays even if the overall JSON is malformed (common)
        if raw_text and (
            not isinstance(raw, dict)
            or not raw
            or (isinstance(raw.get("flashcards"), list) and len(raw.get("flashcards")) == 0)
            or (isinstance(raw.get("quiz"), list) and len(raw.get("quiz")) == 0)
        ):
            salvaged = _salvage_flashcards_quiz_from_malformed(raw_text)
            if salvaged.get("flashcards") or salvaged.get("quiz"):
                if not isinstance(raw, dict):
                    raw = {}
                raw.setdefault("flashcards", [])
                raw.setdefault("quiz", [])
                if isinstance(raw.get("flashcards"), list) and salvaged.get("flashcards"):
                    raw["flashcards"] = salvaged["flashcards"]
                if isinstance(raw.get("quiz"), list) and salvaged.get("quiz"):
                    raw["quiz"] = salvaged["quiz"]
                logger.info(
                    f"[Flashcards/{video_stem}] Salvaged (pass 4): "
                    f"{len(raw.get('flashcards', []))} flashcards, "
                    f"{len(raw.get('quiz', []))} quiz."
                )

        # ── Build typed lists ─────────────────────────────────────────────────
        for card in raw.get("flashcards", []):
            if isinstance(card, dict) and card.get("question"):
                flashcards.append({
                    "question":   card.get("question", ""),
                    "answer":     card.get("answer", ""),
                    "topic":      card.get("topic", ""),
                    "difficulty": card.get("difficulty", "medium"),
                })

        for q in raw.get("quiz", []):
            if isinstance(q, dict) and q.get("question"):
                opts = {k: q[k] for k in ("A", "B", "C", "D") if k in q}
                if not opts and "options" in q:
                    options_val = q["options"]
                    if isinstance(options_val, dict):
                        opts = options_val
                    elif isinstance(options_val, list) and len(options_val) >= 4:
                        opts = {"A": options_val[0], "B": options_val[1], "C": options_val[2], "D": options_val[3]}
                quiz.append({
                    "question":       q.get("question", ""),
                    "options":        opts,
                    "correct_answer": q.get("correct_answer", ""),
                    "explanation":    q.get("explanation", ""),
                    "topic":          q.get("topic", ""),
                })

        logger.info(
            f"[Flashcards/{video_stem}] Done: "
            f"{len(flashcards)} flashcards, {len(quiz)} quiz questions."
        )

        flash_path = write_json_file(
            os.path.join(FLASHCARD_DIR, f"{video_stem}_flashcards.json"),
            flashcards,
        )
        quiz_path = write_json_file(
            os.path.join(QUIZ_DIR, f"{video_stem}_quiz.json"),
            quiz,
        )

        vr["flashcards"]      = flashcards
        vr["flashcards_path"] = flash_path
        vr["quiz"]            = quiz
        vr["quiz_path"]       = quiz_path

        state["state"]           = "done"
        state["flashcard_count"] = len(flashcards)
        state["quiz_count"]      = len(quiz)
        state["error"]           = None

        # ── FIX: ALWAYS persist to DB, even if lists are empty ────────────────
        if user_id:
            try:
                from db_actions import save_flashcards_to_db
                save_flashcards_to_db(user_id, video_stem, flashcards, quiz)
                logger.info(
                    f"[Flashcards/{video_stem}] Persisted to DB: "
                    f"{len(flashcards)} flashcards, {len(quiz)} quiz questions."
                )
            except Exception as db_exc:
                # DB failure must NOT break the in-memory result
                logger.error(
                    f"[Flashcards/{video_stem}] DB persist failed (non-fatal): {db_exc}",
                    exc_info=True,
                )

    except Exception as exc:
        logger.error(
            f"[Flashcards/{video_stem}] Generation FAILED: {exc}",
            exc_info=True,
        )
        state["state"] = "failed"
        state["error"] = str(exc)


# ──────────────────────────────────────────────────────────────────────────────
#  CORE ASYNC PIPELINE  (v3.1.0 / v3.2.0 — 100 % unchanged)
# ──────────────────────────────────────────────────────────────────────────────

async def run_academic_pipeline(
    video_paths: List[str],
    video_fps_overrides: Optional[Dict[str, float]] = None,
    user_id: Optional[str] = None,          # v3.2.1: for DB persist after Phase 2
) -> None:
    """
    Full v3.2.0 pipeline.  user_id is the only addition — it is passed
    straight through to save_lecture_to_db after Phase 2 completes.
    Every other line is identical to v3.2.0.
    """
    global pipeline_running, _pipeline_running_ref, academic_results, stream_frame_counters, _shared_llm, pipeline_status, pipeline_progress

    pipeline_running = _pipeline_running_ref[0] = True
    pipeline_progress = 0
    pipeline_status = "Initializing Academic Pipeline..."

    device = setup_device()

    sources = {f"stream_{i}": p for i, p in enumerate(video_paths)}

    import concurrent.futures
    import threading
    sequential_mode = _LOW_VRAM_SEQUENTIAL

    # Force OCR to CPU on 4GB GPUs to save VRAM for Phi-3
    ocr_use_gpu = False 
    if torch.cuda.is_available():
        logger.info(f"[Memory Optimization] Forcing EasyOCR to CPU (VRAM={torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB detected)")
    else:
        logger.info("[Memory Optimization] CUDA unavailable; running CPU-safe path.")
    logger.info(f"[Pipeline] Mode: {'sequential low-VRAM' if sequential_mode else 'concurrent'}")
    _log_vram("pipeline start")

    llm:           Any = None
    ocr_extractor: Any = None
    _models_ready  = threading.Event()
    _model_error:  List[Optional[Exception]] = [None]

    _ocr_thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=int(os.environ.get("OCR_WORKERS", "2")),
        thread_name_prefix="ocr_worker",
    )

    def _load_models() -> None:
        nonlocal llm, ocr_extractor
        try:
            llm = get_or_load_shared_llm("Stream Pipeline")

            logger.info(f"Loading EasyOCR (gpu={ocr_use_gpu})…")
            ocr_extractor = OCRExtractor(use_gpu=ocr_use_gpu)

            if ocr_extractor is None:
                raise RuntimeError("OCRExtractor returned None.")
            if not callable(getattr(ocr_extractor, "extract", None)):
                raise RuntimeError("OCRExtractor does not expose .extract().")

            logger.info("Models ready [OK] (Phi-3 + EasyOCR)")
            _log_vram("after Phi-3 + EasyOCR load")

        except Exception as exc:
            _model_error[0] = exc
            logger.error(f"Model loading FAILED: {exc}", exc_info=True)
        finally:
            _models_ready.set()

    # We will start the model loader thread, but it will WAIT for Whisper to be
    # released to avoid VRAM exhaustion on 4GB GPUs (RTX 3050).
    # Previous bug: Phase 1 done → Phi-3 load starts while Whisper still holds
    # ~300MB VRAM → peak allocation during 4-bit quantisation kills the process.
    # Fix: wait for _whisper_released_event (set AFTER release_whisper_models()).
    _phase1_done_event = threading.Event()
    _whisper_released_event = threading.Event()
    
    def _load_models_when_ready() -> None:
        logger.info("[Memory] Waiting for Whisper VRAM release before loading Phi-3...")
        _whisper_released_event.wait()
        time.sleep(0.5)  # Let CUDA cache flush settle
        _load_models()

    _model_thread = None
    if not sequential_mode:
        _model_thread = threading.Thread(target=_load_models_when_ready, daemon=True, name="model_loader")
        _model_thread.start()

    _save_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="jpeg_save"
    )
    _executor = None if sequential_mode else concurrent.futures.ThreadPoolExecutor(
        max_workers=len(video_paths),
        thread_name_prefix="whisper",
    )

    def _run_whisper(vp: str) -> Dict:
        try:
            audio_path = academic_results[vp].get("_audio_path")
        except KeyError:
            logger.error(f"[Whisper/bg] academic_results missing for {vp}.")
            return {}
        if not audio_path or not os.path.isfile(audio_path):
            return {}
        try:
            result = transcribe(
                audio_path,
                language   = None,
                model_size = _WHISPER_MODEL_SIZE,
            )
            logger.info(
                f"[Whisper/bg] Done for {Path(vp).stem}: "
                f"{len(result.get('segments', []))} segments "
                f"(model={_WHISPER_MODEL_SIZE})"
            )
            return result
        except Exception as exc:
            logger.error(f"[Whisper/bg] Failed for {vp}: {exc}")
            return {"error": str(exc)}

    whisper_futures: Dict[str, concurrent.futures.Future] = {}
    whisper_results: Dict[str, Dict] = {}
    if sequential_mode:
        logger.info(
            f"[Pipeline] Sequential execution enabled: frame extraction -> Whisper -> release -> Phi-3/OCR -> Phase 2"
        )
    else:
        whisper_futures = {
            vp: _executor.submit(_run_whisper, vp)
            for vp in video_paths
        }
        logger.info(
            f"Started concurrently: frame extraction + model loading + "
            f"Whisper ({_WHISPER_MODEL_SIZE})."
        )

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
    _OCR_BATCH = int(os.environ.get("OCR_BATCH_SIZE", "8"))
    _pending_ocr:      List[tuple] = []
    _pre_model_buffer: List[tuple] = []

    def _flush_ocr_batch_sync(batch: List[tuple]) -> None:
        if not batch:
            return
        if _model_error[0] is not None:
            raise RuntimeError(f"Model loading failed: {_model_error[0]}")
        if ocr_extractor is None:
            raise RuntimeError("ocr_extractor is None.")

        raw_frames = [item[0] for item in batch]
        try:
            all_ocr = ocr_extractor.batch_extract(raw_frames, config.ocr_confidence_threshold)
        except Exception as exc:
            logger.warning(f"Batch OCR failed ({exc}), falling back to sequential.")
            all_ocr = [
                ocr_extractor.extract(f, config.ocr_confidence_threshold)
                for f in raw_frames
            ]

        for (
            frame, frame_id, timestamp, stream_id, video_path, rel_path, frame_url
        ), raw_ocr in zip(batch, all_ocr):
            ocr_text        = ocr_to_text(raw_ocr)
            word_char_count = len(re.findall(r'[A-Za-z0-9]', ocr_text))

            if word_char_count < config.min_ocr_word_chars:
                academic_content = {"importance": "low"}
            elif word_char_count < 50:
                academic_content = {"importance": "medium", "key_concepts": [], "formulas": []}
            else:
                academic_content = {
                    "importance":      "high",
                    "slide_title":     "",
                    "key_concepts":    [],
                    "formulas":        [],
                    "bullet_points":   [],
                    "content_summary": ocr_text[:200],
                }

            _store_frame(
                frame_id, timestamp, stream_id, video_path,
                rel_path, frame_url, ocr_text, academic_content,
            )

    async def _flush_ocr_batch_async(batch: List[tuple]) -> None:
        if not batch:
            return
        if not _models_ready.is_set():
            logger.debug(f"[OCR] Models not ready — buffering {len(batch)} frame(s).")
            _pre_model_buffer.extend(batch)
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_ocr_thread_pool, _flush_ocr_batch_sync, batch)

    def _store_frame(
        frame_id, timestamp, stream_id, video_path,
        rel_path, frame_url, ocr_text, academic_content,
    ) -> None:
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

    async def _drain_pre_model_buffer() -> None:
        if not _pre_model_buffer:
            return
        logger.info(f"[OCR] Draining {len(_pre_model_buffer)} pre-model-load frame(s).")
        for i in range(0, len(_pre_model_buffer), _OCR_BATCH):
            chunk = _pre_model_buffer[i : i + _OCR_BATCH]
            try:
                await _flush_ocr_batch_async(chunk)
            except Exception as exc:
                logger.error(f"[OCR] Pre-model buffer flush failed: {exc}")
        _pre_model_buffer.clear()

    try:
        pipeline_status = "Phase 1: Extracting key frames from videos..."
        pipeline_progress = 5
        total_streams = len(stream_id_to_path)
        for stream_id, video_path in stream_id_to_path.items():
            if not pipeline_running:
                break
            stream_num = int(stream_id.split("_")[-1]) + 1
            _log_pipeline_progress(
                "[Phase 1] Video",
                stream_num - 1,
                total_streams,
                extra=f"extracting frames for {Path(video_path).name}",
            )
            pipeline_status = f"Phase 1: Analyzing frames for '{Path(video_path).name}' ({stream_num}/{total_streams})"
            pipeline_progress = 10 + int((stream_num - 1) / total_streams * 20)

            stats = slide_stats[stream_id]
            sample_timestamps = _get_fixed_sample_timestamps(_get_video_duration_sec(video_path))
            sampled_frames = _extract_fixed_frames(video_path, sample_timestamps)

            stats["frames_seen"] = len(sample_timestamps)
            stats["slides_accepted"] = len(sampled_frames)
            stats["frames_skipped"] = max(0, len(sample_timestamps) - len(sampled_frames))

            if not sampled_frames:
                logger.warning(f"[{stream_id}] No fixed-sample frames extracted for {video_path}.")
                continue

            for frame, timestamp in sampled_frames:
                current_count = stream_frame_counters.get(stream_id, 0)
                if current_count >= len(_FIXED_VIDEO_SAMPLE_POINTS):
                    break

                stream_frame_counters[stream_id] = current_count + 1
                frame_id = stream_frame_counters[stream_id]

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

                _pending_ocr.append(
                    (frame, frame_id, timestamp, stream_id,
                     video_path, rel_path, frame_url)
                )

                if len(_pending_ocr) >= _OCR_BATCH:
                    await _flush_ocr_batch_async(_pending_ocr)
                    _pending_ocr.clear()
                    await asyncio.sleep(0)

            _log_pipeline_progress(
                "[Phase 1] Video",
                stream_num,
                total_streams,
                extra=f"frames={len(sampled_frames)} for {Path(video_path).stem}",
            )

        _completed_normally = True
        logger.info("Fixed-frame extraction complete for all streams.")

        while False and pipeline_running:
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

                current_count = stream_frame_counters.get(stream_id, 0)
                if current_count >= _MAX_FRAMES_EXTRACT:
                    if current_count == _MAX_FRAMES_EXTRACT:
                        logger.warning(
                            f"[{stream_id}] Frame cap reached: {_MAX_FRAMES_EXTRACT} frames."
                        )
                    stats["frames_skipped"] += 1
                    continue

                stats["frames_seen"] += 1

                if not slide_detectors[stream_id].is_new_slide(frame, timestamp):
                    stats["frames_skipped"] += 1
                    logger.debug(f"[{stream_id}] @{timestamp:.1f}s — duplicate, skipped.")
                    continue

                stats["slides_accepted"] += 1
                stream_frame_counters[stream_id] = current_count + 1
                frame_id = stream_frame_counters[stream_id]

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
                if frame_id % 10 == 0:
                    try:
                        write_frames_index(video_path, frame_indices[video_path])
                    except Exception as exc:
                        logger.warning(f"Frame index write: {exc}")

                _pending_ocr.append(
                    (frame, frame_id, timestamp, stream_id,
                     video_path, rel_path, frame_url)
                )

                if len(_pending_ocr) >= _OCR_BATCH:
                    await _flush_ocr_batch_async(_pending_ocr)
                    _pending_ocr.clear()
                    await asyncio.sleep(0)

    finally:
        _phase1_done_event.set() # Signals model loader to start
        if _pending_ocr:
            await _flush_ocr_batch_async(_pending_ocr)
            _pending_ocr.clear()

        if not _completed_normally:
            logger.info("Pipeline stopped early — generating outputs from collected frames.")

        for sid, stats in slide_stats.items():
            vp  = stream_id_to_path.get(sid)
            pct = 100 * stats["frames_skipped"] / max(stats["frames_seen"], 1)
            logger.info(
                f"[{sid}] Slide detection: "
                f"{stats['slides_accepted']} unique slides accepted, "
                f"{stats['frames_skipped']} duplicate frames skipped "
                f"({pct:.0f}% compute saved)"
            )
            if vp and vp in academic_results:
                academic_results[vp]["slide_change_stats"] = stats

        for vp, idx in frame_indices.items():
            if idx:
                try:
                    w = write_frames_index(vp, idx)
                    if vp in academic_results:
                        academic_results[vp]["frames_index_path"] = w
                        academic_results[vp]["frames_index"]      = idx
                except Exception as exc:
                    logger.error(f"Final index write failed: {exc}")

        # ── Whisper results ───────────────────────────────────────────────────
        _outer_timeout = _WHISPER_TIMEOUT_SEC + 120
        if sequential_mode:
            total_videos = len(video_paths)
            pipeline_status = "Phase 1: Transcribing audio with Whisper..."
            for idx, vp in enumerate(video_paths, start=1):
                _log_pipeline_progress(
                    "[Whisper] Video",
                    idx - 1,
                    total_videos,
                    extra=f"starting {Path(vp).name}",
                )
                pipeline_status = f"Phase 1: Transcribing audio for '{Path(vp).name}' ({idx}/{total_videos})"
                pipeline_progress = 30 + int((idx - 1) / total_videos * 20)
                _log_vram(f"before Whisper {idx}/{total_videos}")
                try:
                    whisper_results[vp] = _run_whisper(vp)
                except Exception as exc:
                    logger.error(f"Whisper sequential run failed for {vp}: {exc}")
                    whisper_results[vp] = {}
                _log_pipeline_progress(
                    "[Whisper] Video",
                    idx,
                    total_videos,
                    extra=f"segments={len(whisper_results.get(vp, {}).get('segments', []))}",
                )
                _log_vram(f"after Whisper {idx}/{total_videos}")
        else:
            whisper_results = {}
            for vp, future in whisper_futures.items():
                try:
                    whisper_results[vp] = future.result(timeout=_outer_timeout)
                except concurrent.futures.TimeoutError:
                    logger.error(f"[Whisper] Outer timeout ({_outer_timeout}s) for {vp}.")
                    whisper_results[vp] = {"error": "timeout", "text": "", "segments": []}
                except Exception as exc:
                    logger.error(f"Whisper future failed for {vp}: {exc}")
                    whisper_results[vp] = {}
            _executor.shutdown(wait=False)
        _save_executor.shutdown(wait=True)

        # ── v3.2.3: Release Whisper VRAM before Phi-3 loads ───────────────
        try:
            pipeline_status = "Memory Optimization: Swapping models in VRAM..."
            pipeline_progress = 50
            _log_vram("before Whisper release")
            release_whisper_models()

            logger.info("[Memory] Waiting for VRAM reclamation...")
            await asyncio.sleep(5.0)  # increase from 2.0 to 5.0 for long videos

            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # second sync to confirm flush
                vfree, vtotal = torch.cuda.mem_get_info()
                logger.info(f"[Memory] After 5s cooldown: {vfree/(1024**3):.2f}GB free of {vtotal/(1024**3):.2f}GB")

                # Safety check — abort Phi-3 load if not enough VRAM (only if not already loaded)
                if _shared_llm is None and vfree < 2.8 * (1024**3):
                    logger.error(f"[Memory] Insufficient VRAM ({vfree/(1024**3):.2f}GB) — Phi-3 load would OOM. Aborting.")
                    raise RuntimeError("Insufficient VRAM after Whisper release.")
            _log_vram("after Whisper release")
        except Exception as exc:
            logger.warning(f"[Memory] Whisper release or VRAM check failed: {exc}")
            if isinstance(exc, RuntimeError) and "Insufficient VRAM" in str(exc):
                raise exc  # Re-raise critical OOM safety abort
        
        pipeline_status = "Loading Phi-3 Reasoner..."
        pipeline_progress = 55
        if sequential_mode:
            logger.info("[Pipeline] Sequential mode — loading Phi-3/EasyOCR only after Whisper release.")
            _load_models()
        else:
            _whisper_released_event.set()  # Signal model loader thread to proceed

        if not _models_ready.is_set():
            logger.info("Waiting for model loading to complete before Phase 2…")
            _models_ready.wait()

        if _model_error[0] is None:
            await _drain_pre_model_buffer()
        _ocr_thread_pool.shutdown(wait=False)

        if _model_error[0] is not None:
            logger.error(
                f"[Phase 2 ABORTED] Model load failed: {_model_error[0]}"
            )
            for vp in video_paths:
                if vp in academic_results:
                    academic_results[vp].update({
                        "error":                 str(_model_error[0]),
                        "total_frames_analysed": 0,
                        "lecture_summary":       {},
                        "audio_topics":          {},
                        "study_notes":           None,
                        "flashcards":            [],
                        "quiz":                  [],
                        "knowledge_graph":       None,
                        "pdf_report_path":       None,
                    })
            pipeline_running = _pipeline_running_ref[0] = False
            return

        # ── PHASE 2: per-video outputs ────────────────────────────────────────
        total_phase2 = len(per_stream_frames)
        pipeline_status = "Phase 2: Generating academic outputs..."
        for phase2_idx, (stream_id, frames_list) in enumerate(per_stream_frames.items(), start=1):
            video_path = stream_id_to_path[stream_id]
            vr         = academic_results[video_path]
            stem       = stem_path(video_path)
            pipeline_status = f"Phase 2: Summarizing and generating notes for '{stem}' ({phase2_idx}/{total_phase2})"
            pipeline_progress = 60 + int((phase2_idx - 1) / total_phase2 * 30)
            _log_pipeline_progress(
                "[Phase 2] Video",
                phase2_idx - 1,
                total_phase2,
                extra=f"starting {stem}",
            )

            logger.info(
                f"Phase 2 — generating academic outputs for: {stem} "
                f"({len(frames_list)} frames collected)"
            )

            if not frames_list:
                vr.update({
                    "total_frames_analysed": 0,
                    "lecture_summary":       {},
                    "audio_topics":          {},
                    "study_notes":           None,
                    "flashcards":            [],
                    "quiz":                  [],
                    "pdf_report_path":       None,
                    "knowledge_graph":       None,
                })
                logger.warning(f"No frames collected for {video_path}.")
                continue

            meaningful_frames = [
                fr for fr in frames_list
                if fr.get("academic_content", {}).get("importance") != "low"
                or fr.get("ocr_text", "").strip()
            ]
            if not meaningful_frames:
                logger.warning(
                    f"[{stem}] ALL {len(frames_list)} frames have empty OCR / importance=low."
                )
                frames_for_phase2 = frames_list
            else:
                logger.info(
                    f"[{stem}] {len(meaningful_frames)}/{len(frames_list)} frames "
                    f"have meaningful OCR content."
                )
                frames_for_phase2 = frames_list

            frames_for_llm = _sample_frames_for_phase2(frames_for_phase2, _PHASE2_MAX_FRAMES)
            if len(frames_for_llm) < len(frames_for_phase2):
                logger.info(
                    f"[{stem}] Phase 2 frame sampling: {len(frames_for_phase2)} → "
                    f"{len(frames_for_llm)} frames (cap={_PHASE2_MAX_FRAMES})."
                )

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

            # ── Call 1: metadata ──────────────────────────────────────────────
            logger.info("Phase 2: 2-call split — Call 1 (metadata)…")

            audio_topics:    Dict[str, Any] = {}
            lecture_summary: Dict[str, Any] = {}
            notes_md   = ""
            meta:       Dict[str, Any] = {}

            try:
                call1_prompt = _patch(
                    prompt_metadata(
                        video_path,
                        frames_for_llm,
                        transcript,
                        sample_n     = min(5, len(frames_for_llm)),
                        max_concepts = 6,
                    )
                )
                logger.info(
                    f"[DEBUG] Call 1 prompt: {len(call1_prompt)} chars "
                    f"(~{len(call1_prompt)//4} tokens)"
                )
                meta = serialize(_llm_reason(llm, call1_prompt, _MAX_TOKENS_META)) or {}

                if not meta:
                    raw_text = getattr(llm, "_last_raw_output", "") or ""
                    logger.warning(f"[DEBUG] Call 1 raw output: {raw_text[:300]!r}")
                    if raw_text:
                        meta = _extract_first_json_object(raw_text) or {}
                        if meta:
                            logger.info(f"Call 1 JSON recovery: {list(meta.keys())}")

            except Exception as exc:
                logger.error(f"Phase 2 Call 1 (metadata) failed: {exc}", exc_info=True)

            if meta:
                lecture_summary = {
                    "lecture_title":     meta.get("lecture_title", ""),
                    "subject_area":      meta.get("subject_area", ""),
                    "main_topics":       meta.get("topics", []),
                    "learning_outcomes": meta.get("learning_outcomes", []),
                    "summary":           meta.get("summary", ""),
                    "difficulty_level":  meta.get("difficulty", ""),
                }
                audio_topics = {
                    "lecture_title":    meta.get("lecture_title", ""),
                    "subject_area":     meta.get("subject_area", ""),
                    "topics_covered":   meta.get("topics", []),
                    "key_concepts":     [
                        {"concept": c, "explanation": ""}
                        for c in meta.get("key_concepts", [])
                    ],
                    "important_points": meta.get("learning_outcomes", []),
                    "summary":          meta.get("summary", ""),
                }
                logger.info(
                    f"Call 1 done: title='{meta.get('lecture_title','?')[:50]}'"
                )
            else:
                logger.warning("Call 1 returned empty metadata — Call 2 will use defaults.")

            # ── Call 2: study notes ───────────────────────────────────────────
            logger.info("Phase 2: 2-call split — Call 2 (study notes)…")
            raw_formulas: List[str] = list({
                f
                for fr in frames_for_llm
                for f in fr.get("academic_content", {}).get("formulas", [])
            })[:4]

            try:
                call2_prompt = _patch(
                    prompt_study_notes_text(
                        lecture_title     = meta.get("lecture_title", stem),
                        subject_area      = meta.get("subject_area", "General"),
                        difficulty        = meta.get("difficulty", ""),
                        topics            = meta.get("topics", []),
                        key_concepts      = meta.get("key_concepts", []),
                        learning_outcomes = meta.get("learning_outcomes", []),
                        summary           = meta.get("summary", ""),
                        formulas          = raw_formulas,
                    )
                )
                logger.info(
                    f"[DEBUG] Call 2 prompt: {len(call2_prompt)} chars "
                    f"(~{len(call2_prompt)//4} tokens)"
                )
                notes_md = _llm_reason_text(llm, call2_prompt, _MAX_TOKENS_NOTES)
                logger.info(f"Call 2 done: notes={len(notes_md)} chars")

            except Exception as exc:
                logger.error(f"Phase 2 Call 2 (notes) failed: {exc}", exc_info=True)

            if not notes_md or not notes_md.strip():
                title    = meta.get("lecture_title", stem)
                summary  = meta.get("summary", "")
                topics   = meta.get("topics", [])
                concepts = meta.get("key_concepts", [])
                outcomes = meta.get("learning_outcomes", [])
                lines    = [f"# Study Notes: {title}", ""]
                if summary:
                    lines += ["## Overview", "", summary, ""]
                if topics:
                    lines += ["## Topics", ""] + [f"- {t}" for t in topics] + [""]
                if concepts:
                    lines += ["## Key Concepts", ""] + [f"- {c}" for c in concepts] + [""]
                if outcomes:
                    lines += ["## Learning Outcomes", ""] + [f"- {o}" for o in outcomes] + [""]
                notes_md = "\n".join(lines)
                logger.info("Call 2: empty output — built minimal notes from metadata.")
            elif not notes_md.lstrip().startswith("#"):
                notes_md = f"# Study Notes: {meta.get('lecture_title', stem)}\n\n" + notes_md

            vr["audio_topics"]          = audio_topics
            vr["lecture_summary"]       = lecture_summary
            vr["total_frames_analysed"] = len(frames_list)

            # ── Deduplication ─────────────────────────────────────────────────
            audio_concept_frames: List[Dict] = []
            for c in audio_topics.get("key_concepts", []):
                name = (
                    (c.get("concept") or c.get("name") or "")
                    if isinstance(c, dict)
                    else str(c)
                )
                if name.strip():
                    audio_concept_frames.append(
                        {"academic_content": {"key_concepts": [name.strip()]}}
                    )
            all_frames_for_dedup = frames_list + audio_concept_frames

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as _dedup_pool:
                f_concepts = _dedup_pool.submit(
                    _deduplicator.deduplicate_concepts, all_frames_for_dedup
                )
                f_defs = _dedup_pool.submit(
                    _deduplicator.deduplicate_definitions, frames_list
                )
                f_formulas = _dedup_pool.submit(
                    _deduplicator.deduplicate_formulas, frames_list
                )
                deduped_concepts = f_concepts.result()
                deduped_defs     = f_defs.result()
                deduped_formulas = f_formulas.result()

            vr["deduped_concepts"] = deduped_concepts
            vr["deduped_formulas"] = deduped_formulas
            logger.info(
                f"Dedup ({_deduplicator.backend}): "
                f"{len(deduped_concepts)} concepts, {len(deduped_formulas)} formulas."
            )

            if not notes_md:
                notes_md = f"# Study Notes: {stem}\n\nNotes generation failed."
            notes_path = write_text_file(
                os.path.join(NOTES_DIR, f"{stem}_study_notes.md"), notes_md
            )
            vr["study_notes"]      = notes_md
            vr["study_notes_path"] = notes_path

            vr["flashcards"]      = []
            vr["flashcards_path"] = None
            vr["quiz"]            = []
            vr["quiz_path"]       = None

            _flashcard_states[stem] = {
                "state":           "idle",
                "flashcard_count": 0,
                "quiz_count":      0,
                "error":           None,
            }

            def _build_graph():
                try:
                    graph      = _graph_builder.build(frames_list, audio_topics, lecture_summary)
                    graph_d3   = _graph_builder.to_d3_json(graph)
                    graph_path = _graph_builder.save(
                        graph,
                        os.path.join(GRAPH_DIR, f"{stem}_knowledge_graph.json"),
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
                        flashcards      = [],
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

            # ── v3.2.1: persist lecture to DB ─────────────────────────────────
            logger.info(
                f"[{stem}] Pipeline complete. "
                f"Call POST /generate/flashcards/{stem} when ready."
            )
            if user_id:
                from db_actions import save_pipeline_result_to_db
                save_pipeline_result_to_db(user_id, video_path, vr, stem)

        pipeline_status = "Pipeline Complete"
        pipeline_progress = 100
        pipeline_running = _pipeline_running_ref[0] = False


# ──────────────────────────────────────────────────────────────────────────────
#  PYDANTIC MODELS
# ──────────────────────────────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    confidence: int           = Field(..., ge=1, le=5)
    correct:    bool          = Field(False)
    session_id: Optional[str] = Field(None)


class QuizSessionRequest(BaseModel):
    total_questions: int
    correct_answers: int


# v3.2.1: auth models
class UserRegister(BaseModel):
    username: str
    email:    str
    password: str


class Token(BaseModel):
    access_token: str
    token_type:   str


# ──────────────────────────────────────────────────────────────────────────────
#  v3.2.1: AUTH ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/auth/register", tags=["Auth"], summary="Register a new user account")
async def register(body: UserRegister):
    username = body.username.strip()
    if not username or not body.password:
        raise HTTPException(400, "Username and password are required.")
    
    from database_v2 import SessionLocal, User as DBUser
    db = SessionLocal()
    existing = db.query(DBUser).filter(DBUser.username == username).first()
    if existing:
        db.close()
        raise HTTPException(400, "Username already exists.")
    new_user = DBUser(username=username, email=body.email, password_hash=auth.get_password_hash(body.password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    user_id_str = str(new_user.id)
    db.close()

    access_token = auth.create_access_token(
        data={"sub": username, "email": body.email, "uid": user_id_str}
    )
    return {
        "access_token": access_token,
        "token_type":   "bearer",
        "user":         {"id": user_id_str, "username": username, "email": body.email},
    }


@app.post("/auth/login", tags=["Auth"], summary="Login and receive a JWT token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    username = form_data.username.strip()
    if not username or not form_data.password:
        raise HTTPException(
            401,
            "Incorrect username or password",
            {"WWW-Authenticate": "Bearer"},
        )
    from database_v2 import SessionLocal, User as DBUser
    db = SessionLocal()
    user = db.query(DBUser).filter(DBUser.username == username).first()
    db.close()
    
    if not user or not auth.verify_password(form_data.password, user.password_hash):
        raise HTTPException(401, "Incorrect username or password")

    access_token = auth.create_access_token(data={"sub": username, "uid": str(user.id)})
    return {
        "access_token": access_token,
        "token_type":   "bearer",
        "user":         {"id": str(user.id), "username": username},
    }


@app.get("/auth/me", tags=["Auth"], summary="Return the currently authenticated user")
async def me(current_user: User = Depends(auth.get_current_user)):
    return {"id": current_user.id, "username": current_user.username, "email": current_user.email}


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

@app.post("/chat", summary="Personalized Chatbot", tags=["Chat"])
async def chat_endpoint(
    req: ChatRequest,
    current_user: User = Depends(auth.get_current_user)
):
    global _shared_llm
    from db_actions import get_db_media_states
    
    # Get user media context
    media_states = get_db_media_states(current_user.id)
    
    context_lines = []
    if not media_states:
        context_lines.append("The user has not uploaded any media yet.")
    else:
        for stem, info in media_states.items():
            title = info.get("display_name") or stem
            m_type = info.get("input_type", "unknown")
            notes_ready = "Yes" if info.get("study_notes_ready") else "No"
            cards_ready = "Yes" if info.get("flashcards_ready") else "No"
            context_lines.append(f"- Title: {title} (Type: {m_type}), Notes Ready: {notes_ready}, Flashcards Ready: {cards_ready}")
            
    context_str = "\n".join(context_lines)
    
    sys_prompt = f"""You are EDUvance's AI Assistant created by Mr.Rishi Singh. Your ONLY purpose is to answer questions about how the app works and provide information about the user's uploaded media and academic content.

Here is the user's uploaded content summary:
{context_str}

How the app works: Users upload lectures (video, audio, or slide images/PDFs). The AI transcribes, summarizes, and generates study notes, flashcards, and quizzes.

RESPONSE RULES:
- If the input is ONLY a greeting (e.g. "hi", "hello", "hey"), respond with a friendly greeting and ask how you can help — nothing more.
- Keep all answers short and to the point (2–4 sentences max).
- If asked to quiz the user, generate a relevant multiple-choice or short-answer question based on their uploaded content. After they answer, give brief feedback and optionally ask another.
- If the question is outside the app's functionality or the user's uploaded content, politely decline in one sentence.
- Never over-explain. Be concise, helpful, and academic in tone."""

    _shared_llm = get_or_load_shared_llm("Chat")

    # Build prompt for Phi-3 Instruct format
    conversation_prompt = f"<|system|>\n{sys_prompt}<|end|>\n"
    for m in req.messages[-5:]:  # keeping last 5 messages to avoid blowing context
        role = "user" if m.role == "user" else "assistant"
        conversation_prompt += f"<|{role}|>\n{m.content}<|end|>\n"
    
    conversation_prompt += "<|assistant|>\n"
    
    try:
        response_text = _llm_reason_text(_shared_llm, conversation_prompt, 600)
        # Clear out any trailing <|end|> or <|endoftext|> if the model outputs it
        response_text = response_text.replace("<|end|>", "").replace("<|endoftext|>", "").strip()
        return {"response": response_text}
    except Exception as e:
        logger.error(f"[Chat] failed: {e}")
        raise HTTPException(500, f"Error generating response: {e}")


# ──────────────────────────────────────────────────────────────────────────────
#  UPLOAD ENDPOINTS  (pipeline logic unchanged; auth + DB added)
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/upload/video", summary="Upload 1–3 lecture videos", tags=["Upload"])
async def upload_video(
    file1: UploadFile = File(None),
    file2: UploadFile = File(None),
    file3: UploadFile = File(None),
    current_user: User = Depends(auth.get_current_user),   # v3.2.1
) -> JSONResponse:
    global pipeline_task, pipeline_running

    uploaded = [f for f in (file1, file2, file3) if f is not None]
    if not uploaded:
        raise HTTPException(400, "At least one video file is required.")
    for f in uploaded:
        _assert_ext(f.filename, VIDEO_EXTENSIONS, "video")

    if pipeline_task and not pipeline_task.done():
        pipeline_running = _pipeline_running_ref[0] = False
        try:
            await asyncio.wait_for(pipeline_task, timeout=5.0)
        except asyncio.TimeoutError:
            pipeline_task.cancel()

    # v3.2.1: Clear ONLY current user's in-memory video results to allow other users to keep theirs
    for key in [
        k for k, v in academic_results.items()
        if v.get("input_type") == "video" and v.get("user_id") == current_user.id
    ]:
        del academic_results[key]

    video_paths: List[str] = []
    video_fps_overrides: Dict[str, float] = {}

    for upload in uploaded:
        # v3.2.1: prefix filename with user id to avoid collisions between users
        safe_name = f"{current_user.id}_{upload.filename}"
        dest = os.path.join(UPLOAD_DIR, safe_name)
        with open(dest, "wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        logger.info(f"Video saved locally → {dest}  (user={current_user.username})")

        # Upload to MinIO
        minio_object_name = f"videos/{safe_name}"
        try:
            minio_client.upload_file(dest, minio_object_name)
            logger.info(f"Video uploaded to MinIO → {minio_object_name}")
        except Exception as e:
            logger.error(f"Failed to upload video to MinIO: {e}")

        meta       = extract_video_metadata(dest)
        audio_path = extract_audio(dest, AUDIO_DIR)
        frames_dir = _video_frames_dir(dest)

        duration_sec = _get_video_duration_sec(dest)
        adaptive_fps = _adaptive_fps(duration_sec)

        if duration_sec is not None:
            duration_min = duration_sec / 60
            logger.info(
                f"[{Path(dest).stem}] Duration: {duration_min:.1f} min "
                f"→ adaptive FPS: {adaptive_fps:.2f}"
            )
            if duration_sec > 3600:
                logger.warning(
                    f"[{Path(dest).stem}] Video longer than 60 min "
                    f"({duration_min:.1f} min). Capped at {_MAX_FRAMES_EXTRACT} frames."
                )
        else:
            logger.info(f"[{Path(dest).stem}] Duration unknown — using FPS {adaptive_fps:.2f}")

        video_fps_overrides[dest] = adaptive_fps
        meta["duration_sec"]      = duration_sec
        meta["adaptive_fps"]      = adaptive_fps

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
            "duration_sec":          duration_sec,
            "adaptive_fps":          adaptive_fps,
            "whisper_model":         _WHISPER_MODEL_SIZE,
            # v3.2.1
            "user_id":               current_user.id,
            "created_at":            time.time(),
        }
        video_paths.append(dest)

    # Keep strong reference to task to prevent garbage collection mid-execution
    pipeline_task = asyncio.create_task(
        run_academic_pipeline(
            video_paths,
            video_fps_overrides=video_fps_overrides,
            user_id=current_user.id,            # v3.2.1
        )
    )
    _background_tasks.add(pipeline_task)
    pipeline_task.add_done_callback(_background_tasks.discard)

    duration_info = {}
    for vp in video_paths:
        vr = academic_results[vp]
        dur = vr.get("duration_sec")
        duration_info[Path(vp).name] = {
            "duration_min":  round(dur / 60, 1) if dur else "unknown",
            "adaptive_fps":  vr.get("adaptive_fps", config.fps),
            "max_frames":    _MAX_FRAMES_EXTRACT,
            "whisper_model": _WHISPER_MODEL_SIZE,
        }

    return JSONResponse({
        "message": (
            f"Academic pipeline v3.2.1 started for {len(video_paths)} video(s). "
            "Notes and PDF will be generated automatically. "
            "Flashcards and quiz require POST /generate/flashcards/{stem}."
        ),
        "videos":       video_paths,
        "video_config": duration_info,
        "poll":         "GET /status",
        "outputs": {
            "json":            "GET /results/video",
            "study_notes":     "GET /results/notes/{stem}",
            "pdf":             "GET /results/pdf/{stem}",
            "flashcards":      "POST /generate/flashcards/{stem}  ← trigger first",
            "flashcards_get":  "GET  /results/flashcards/{stem}   ← retrieve after",
            "quiz_get":        "GET  /results/quiz/{stem}",
            "knowledge_graph": "GET  /results/graph/{stem}",
            "frames":          "GET  /results/frames/{stem}",
        },
    })


@app.post("/upload/image", summary="Upload 1–3 slide or diagram images", tags=["Upload"])
async def upload_image(
    file1:    UploadFile = File(None),
    file2:    UploadFile = File(None),
    file3:    UploadFile = File(None),
    language: str        = Query("en"),
    current_user: User   = Depends(auth.get_current_user),
) -> JSONResponse:
    global pipeline_task
    uploaded = [f for f in (file1, file2, file3) if f is not None]
    if not uploaded:
        raise HTTPException(400, "At least one image file is required.")
    for f in uploaded:
        _assert_ext(f.filename, IMAGE_EXTENSIONS, "image")

    # Clear previous in-memory results for images for this user
    for key in [
        k for k, v in academic_results.items()
        if v.get("input_type") == "images" and v.get("user_id") == current_user.id
    ]:
        del academic_results[key]

    lang_info = _lang_detector.from_code(language)
    image_paths = []
    
    # Use a consistent stem for the whole batch
    batch_ts   = int(time.time())
    batch_name = uploaded[0].filename.rsplit('.', 1)[0]
    batch_stem = f"{current_user.id}_{batch_name}_{batch_ts}_img"

    for upload in uploaded:
        safe_name = f"{current_user.id}_{batch_ts}_{upload.filename}"
        dest = os.path.join(IMAGE_DIR, safe_name)
        with open(dest, "wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        image_paths.append(dest)
        
        # Upload to MinIO
        minio_object_name = f"images/{safe_name}"
        try:
            minio_client.upload_file(dest, minio_object_name)
            logger.info(f"Image uploaded to MinIO → {minio_object_name}")
        except Exception as e:
            logger.error(f"Failed to upload image to MinIO: {e}")

    # Initialize result record for image batches
    academic_results[batch_stem] = {
        "input_type": "images",
        "user_id":    current_user.id,
        "created_at": time.time(),
    }

    pipeline_task = asyncio.create_task(
        run_image_batch_pipeline(
            image_paths,
            user_id=current_user.id,
            batch_stem=batch_stem
        )
    )
    _flashcard_states[batch_stem] = {
        "state":           "idle",
        "flashcard_count": 0,
        "quiz_count":      0,
        "error":           None,
    }

    return JSONResponse({
        "message":    f"{len(image_paths)} image(s) received. Full analysis pipeline started.",
        "batch_stem": batch_stem,
        "poll":       "GET /status",
        "outputs": {
            "study_notes":        f"GET /results/notes/{batch_stem}",
            "pdf":                f"GET /results/pdf/{batch_stem}",
            "knowledge_graph":    f"GET /results/graph/{batch_stem}",
            "flashcards_trigger": f"POST /generate/flashcards/{batch_stem}",
            "flashcards_get":     f"GET /results/flashcards/{batch_stem}",
            "quiz_get":           f"GET /results/quiz/{batch_stem}",
            "frames":             f"GET /results/frames/{batch_stem}",
        },
    })


@app.post("/upload/audio", summary="Upload 1–3 lecture audio files", tags=["Upload"])
async def upload_audio(
    file1: UploadFile = File(None),
    file2: UploadFile = File(None),
    file3: UploadFile = File(None),
    current_user: User = Depends(auth.get_current_user),     # v3.2.1
) -> JSONResponse:
    uploaded = [f for f in (file1, file2, file3) if f is not None]
    if not uploaded:
        raise HTTPException(400, "At least one audio file is required.")
    for f in uploaded:
        _assert_ext(f.filename, AUDIO_EXTENSIONS, "audio")

    # Clear previous in-memory results for audio for this user
    for key in [
        k for k, v in academic_results.items()
        if v.get("input_type") == "audio" and v.get("user_id") == current_user.id
    ]:
        del academic_results[key]

    results = []
    for upload in uploaded:
        safe_name = f"{current_user.id}_{upload.filename}"
        dest = os.path.join(AUDIO_DIR, safe_name)
        with open(dest, "wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        
        # Upload to MinIO
        minio_object_name = f"audio/{safe_name}"
        try:
            minio_client.upload_file(dest, minio_object_name)
            logger.info(f"Audio uploaded to MinIO → {minio_object_name}")
        except Exception as e:
            logger.error(f"Failed to upload audio to MinIO: {e}")

        ext             = Path(dest).suffix.lower()
        processing_path = convert_to_wav(dest, AUDIO_DIR) if ext != ".wav" else dest

        try:
            transcription = transcribe(
                processing_path,
                language   = None,
                model_size = _WHISPER_MODEL_SIZE,
            )
            lang_info = _lang_detector.from_whisper(transcription)
            entry = {
                "input_type":        "audio",
                "audio_path":        dest,
                "detected_language": {"code": lang_info["code"], "name": lang_info["name"]},
                "transcription":     transcription,
                "whisper_model":     _WHISPER_MODEL_SIZE,
                "user_id":           current_user.id,
            }
        except Exception as exc:
            logger.error(f"Audio processing failed: {exc}")
            entry = {"input_type": "audio", "audio_path": dest, "error": str(exc)}

        academic_results[dest] = entry
        results.append(entry)
        
        # v3.3.0: Persist direct audio uploads to DB immediately
        if current_user.id:
            from db_actions import save_pipeline_result_to_db
            # For audio, batch_stem is just the filename stem
            stem = Path(dest).stem
            save_pipeline_result_to_db(str(current_user.id), dest, entry, stem)
            logger.info(f"[Audio] Persisted transription for '{stem}' to DB.")

    return JSONResponse({
        "message":           f"{len(results)} audio file(s) transcribed.",
        "results":           results,
        "whisper_available": WHISPER_AVAILABLE,
        "whisper_model":     _WHISPER_MODEL_SIZE,
        "note":              "Language is auto-detected by Whisper.",
    })


@app.post("/upload/document", summary="Upload 1–3 PDF lecture-note files", tags=["Upload"])
async def upload_document(
    file1: UploadFile = File(None),
    file2: UploadFile = File(None),
    file3: UploadFile = File(None),
    current_user: User = Depends(auth.get_current_user),
) -> JSONResponse:
    """
    Full pipeline for PDF lecture notes — identical quality to video / image uploads.

    Each PDF page is treated as a "frame":
      Phase 1  — page rendering (PyMuPDF) + text extraction (pdfplumber) + OCR + LLM
      Phase 2  — metadata (Call 1) + study notes (Call 2)
      Phase 3  — knowledge graph + PDF academic report
      On-demand — POST /generate/flashcards/{stem} for flashcards + MCQ quiz

    Requires:  pip install pymupdf pdfplumber
    """
    global pipeline_task, pipeline_running, _pipeline_running_ref

    uploaded = [f for f in (file1, file2, file3) if f is not None]
    if not uploaded:
        raise HTTPException(400, "At least one document file is required.")

    # Only PDF is fully supported; .txt is processed as a single-page PDF fallback
    SUPPORTED = {".pdf", ".txt"}
    for f in uploaded:
        ext = Path(f.filename).suffix.lower()
        if ext not in SUPPORTED:
            raise HTTPException(
                415,
                f"'{f.filename}' — only PDF and plain-text files are supported for "
                f"full pipeline processing. Got: {ext}"
            )

    # Check that at least one PDF backend is available before accepting the upload
    backends = _check_pdf_backends()
    if not backends.get("pymupdf") and not backends.get("pdfplumber"):
        raise HTTPException(
            503,
            "PDF processing backends not installed on the server. "
            "Ask your administrator to run: pip install pymupdf pdfplumber"
        )

    results_meta = []
    tasks_started = []

    for upload in uploaded:
        safe_name  = f"{current_user.id}_{upload.filename}"
        dest       = os.path.join(UPLOAD_DIR, safe_name)

        with open(dest, "wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        logger.info(f"Document saved locally → {dest}  (user={current_user.username})")

        # Upload to MinIO (same as video/image)
        minio_object_name = f"documents/{safe_name}"
        try:
            minio_client.upload_file(dest, minio_object_name)
            logger.info(f"Document uploaded to MinIO → {minio_object_name}")
        except Exception as e:
            logger.error(f"Failed to upload document to MinIO: {e}")

        # Build a unique stem for this document
        batch_ts   = int(time.time())
        clean_name = Path(upload.filename).stem
        batch_stem = f"{current_user.id}_{clean_name}_{batch_ts}_doc"

        # Register flashcard state early so /status is consistent
        _flashcard_states[batch_stem] = {
            "state":           "idle",
            "flashcard_count": 0,
            "quiz_count":      0,
            "error":           None,
        }

        # Launch the pipeline as a background async task
        _pipeline_running_ref[0] = pipeline_running = True

        task = asyncio.create_task(
            run_pdf_pipeline(
                pdf_path              = dest,
                user_id               = current_user.id,
                batch_stem            = batch_stem,
                shared_llm            = _shared_llm,         # reuse if already loaded
                academic_results      = academic_results,
                flashcard_states      = _flashcard_states,
                pipeline_running_ref  = _pipeline_running_ref,
                shared_llm_ref        = _shared_llm_ref,     # pass mutable ref for PDF pipeline to write back LLM
            )
        )
        pipeline_task = task
        tasks_started.append(task)

        results_meta.append({
            "filename":   upload.filename,
            "batch_stem": batch_stem,
            "outputs": {
                "status":             "GET /status",
                "study_notes":        f"GET /results/notes/{batch_stem}",
                "pdf_report":         f"GET /results/pdf/{batch_stem}",
                "knowledge_graph":    f"GET /results/graph/{batch_stem}",
                "frames_pages":       f"GET /results/frames/{batch_stem}",
                "flashcards_trigger": f"POST /generate/flashcards/{batch_stem}",
                "flashcards_get":     f"GET /results/flashcards/{batch_stem}",
                "quiz_get":           f"GET /results/quiz/{batch_stem}",
            },
        })

    return JSONResponse({
        "message": (
            f"{len(results_meta)} document(s) received. "
            "Full analysis pipeline started — notes and PDF report will be "
            "generated automatically. Flashcards require "
            "POST /generate/flashcards/{stem}."
        ),
        "backends": backends,
        "documents": results_meta,
        "poll": "GET /status",
    })


# ──────────────────────────────────────────────────────────────────────────────
#  ON-DEMAND FLASHCARD GENERATION  (v3.0.3 + v3.2.1 DB persist)
# ──────────────────────────────────────────────────────────────────────────────

@app.post(
    "/generate/flashcards/{video_stem}",
    summary="Trigger on-demand flashcard + quiz generation from saved notes",
    tags=["Flashcards & Quiz"],
)
async def generate_flashcards(
    video_stem: str,
    current_user: User = Depends(auth.get_current_user),
) -> JSONResponse:
    vr = _find_any_lecture(video_stem, current_user.id)
    if vr is None:
        notes_path = os.path.join(NOTES_DIR, f"{video_stem}_study_notes.md")
        if not os.path.isfile(notes_path):
            raise HTTPException(
                404,
                f"No pipeline result found for '{video_stem}'. Upload the video first.",
            )
    else:
        if not vr.get("study_notes") and not os.path.isfile(
            os.path.join(NOTES_DIR, f"{video_stem}_study_notes.md")
        ):
            raise HTTPException(
                503,
                "Study notes not yet ready. Wait for study_notes_ready=true in GET /status.",
            )

    existing_state = _flashcard_states.get(video_stem, {})
    if existing_state.get("state") == "running":
        return JSONResponse(
            status_code=202,
            content={
                "message": "Generation already in progress.",
                "state":   "running",
                "poll":    f"GET /generate/flashcards/{video_stem}/status",
            },
        )
    if existing_state.get("state") == "done":
        return JSONResponse(
            status_code=200,
            content={
                "message":         "Flashcards already generated.",
                "state":           "done",
                "flashcard_count": existing_state.get("flashcard_count", 0),
                "quiz_count":      existing_state.get("quiz_count", 0),
                "retrieve":        f"GET /results/flashcards/{video_stem}",
            },
        )

    _flashcard_states[video_stem] = {
        "state":           "pending",
        "flashcard_count": 0,
        "quiz_count":      0,
        "error":           None,
    }

    # v3.2.1: ensure user_id is passed but avoid passing request-scoped db session
    async def _gen_with_db():
        if _shared_llm is not None:
            await _run_flashcard_generation(
                video_stem, _shared_llm,
                user_id=current_user.id,
            )
        else:
            await _run_flashcard_generation_with_llm(video_stem, user_id=current_user.id)

    task = asyncio.create_task(_gen_with_db())
    _flashcard_tasks[video_stem] = task

    return JSONResponse(
        status_code=202,
        content={
            "message": f"Flashcard generation started for '{video_stem}'.",
            "state":   "pending",
            "poll":    f"GET /generate/flashcards/{video_stem}/status",
            "retrieve_when_done": {
                "flashcards": f"GET /results/flashcards/{video_stem}",
                "quiz":       f"GET /results/quiz/{video_stem}",
            },
        },
    )


@app.get(
    "/generate/flashcards/{video_stem}/status",
    summary="Poll flashcard/quiz generation progress",
    tags=["Flashcards & Quiz"],
)
async def flashcard_generation_status(
    video_stem: str,
    current_user: User = Depends(auth.get_current_user),    # v3.2.1
) -> JSONResponse:
    state = _flashcard_states.get(video_stem)

    if state is None:
        flash_path = os.path.join(FLASHCARD_DIR, f"{video_stem}_flashcards.json")
        quiz_path  = os.path.join(QUIZ_DIR,       f"{video_stem}_quiz.json")
        if os.path.isfile(flash_path) and os.path.isfile(quiz_path):
            try:
                fc = json.load(open(flash_path, encoding="utf-8"))
                qc = json.load(open(quiz_path,  encoding="utf-8"))
                return JSONResponse({
                    "video_stem":      video_stem,
                    "state":           "done",
                    "flashcard_count": len(fc),
                    "quiz_count":      len(qc),
                    "error":           None,
                    "note":            "Loaded from disk (previous session).",
                    "retrieve": {
                        "flashcards": f"GET /results/flashcards/{video_stem}",
                        "quiz":       f"GET /results/quiz/{video_stem}",
                    },
                })
            except Exception:
                pass

        return JSONResponse({
            "video_stem": video_stem,
            "state":      "idle",
            "message":    f"Call POST /generate/flashcards/{video_stem} to start.",
        })

    response: Dict[str, Any] = {"video_stem": video_stem, **state}
    if state["state"] == "done":
        response["retrieve"] = {
            "flashcards": f"GET /results/flashcards/{video_stem}",
            "quiz":       f"GET /results/quiz/{video_stem}",
        }
    return JSONResponse(response)


# ──────────────────────────────────────────────────────────────────────────────
#  CONTROL  (auth added)
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/stop", summary="Stop the running pipeline", tags=["Control"])
async def stop_pipeline(
    current_user: User = Depends(auth.get_current_user),
) -> JSONResponse:
    global pipeline_running, _pipeline_running_ref
    pipeline_running = _pipeline_running_ref[0] = False
    return JSONResponse({"message": "Stop signal sent."})


@app.delete("/results", summary="Clear all in-memory results", tags=["Control"])
async def clear_results(
    current_user: User = Depends(auth.get_current_user),
) -> JSONResponse:
    global academic_results, _flashcard_states, _flashcard_tasks
    
    keys_to_del = [k for k, v in academic_results.items() if v.get("user_id") == current_user.id]
    for k in keys_to_del:
        del academic_results[k]
        
    stems_to_del = [s for s, state in _flashcard_states.items() if s.split('_', 1)[0] == str(current_user.id)]
    for s in stems_to_del:
        if s in _flashcard_states: del _flashcard_states[s]
        if s in _flashcard_tasks: del _flashcard_tasks[s]
        
    return JSONResponse({"message": f"In-memory results cleared for user {current_user.username}."})


# ──────────────────────────────────────────────────────────────────────────────
#  STATUS  (auth added; DB fallback added for cross-session persistence)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/status", summary="Pipeline progress and per-video readiness", tags=["Status"])
def status(
    current_user: User = Depends(auth.get_current_user),
) -> JSONResponse:
    # In-memory results filtered to current user
    # v3.2.1: treat video and image-batch the same in status
    all_active = {
        p: v for p, v in academic_results.items()
        if v.get("input_type") in ("video", "images", "pdf", "document", "live") and v.get("user_id") == current_user.id
    }
    audios = {
        p: v for p, v in academic_results.items()
        if v.get("input_type") == "audio" and v.get("user_id") == current_user.id
    }

    from db_actions import get_db_media_states
    db_states = get_db_media_states(str(current_user.id))
    db_stems = set(db_states.keys())
    mem_stems = {Path(p).stem for p in all_active}

    video_states = {}
    for p, v in all_active.items():
        ss    = v.get("slide_change_stats", {})
        lang  = v.get("detected_language") or {}
        stem  = Path(p).stem
        fc_st = _flashcard_states.get(stem, {})
        dur   = v.get("duration_sec")
        created_at = v.get("created_at") or time.time()
        v_state = {
            "source":                   "in_memory",
            "created_at":               created_at,
            "duration_min":             round(dur / 60, 1) if dur else None,
            "adaptive_fps":             v.get("adaptive_fps"),
            "whisper_model":            v.get("whisper_model", _WHISPER_MODEL_SIZE),
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
            "flashcards_generation_state": fc_st.get("state", "idle"),
            "flashcards_ready":         fc_st.get("state") == "done" or fc_st.get("flashcard_count", 0) > 0 or (db_states.get(stem, {}).get("flashcard_count", 0) > 0),
            "flashcard_count":          fc_st.get("flashcard_count", 0) or db_states.get(stem, {}).get("flashcard_count", 0),
            "quiz_ready":               fc_st.get("state") == "done" or fc_st.get("quiz_count", 0) > 0 or (db_states.get(stem, {}).get("quiz_count", 0) > 0),
            "quiz_count":               fc_st.get("quiz_count", 0) or db_states.get(stem, {}).get("quiz_count", 0),
            "graph_ready":              v.get("knowledge_graph") is not None,
            "pdf_ready":                v.get("pdf_report_path") is not None,
            "lecture_title":            v.get("lecture_summary", {}).get("lecture_title"),
            "subject_area":             v.get("lecture_summary", {}).get("subject_area"),
            "difficulty":               v.get("lecture_summary", {}).get("difficulty_level"),
            "pipeline_error":           v.get("error"),
            "video_stem":               stem,
            "display_name":             stem.split('_', 1)[-1] if '_' in stem else stem,
            "flashcard_generate_url":   f"POST /generate/flashcards/{stem}",
            "input_type":               v.get("input_type", "video"),
            "frames_index":             v.get("frames_index", []) if v.get("input_type") == "images" else None,
        }

        video_states[os.path.basename(p)] = v_state

    # DB-only lectures (not in memory — e.g. from a previous server session)
    for stem in sorted(db_stems - mem_stems):
        video_states[stem] = db_states[stem]

    return JSONResponse({
        "pipeline_running":       pipeline_running,
        "pipeline_status":        pipeline_status,
        "pipeline_progress":      pipeline_progress,
        "task_done":              pipeline_task.done() if pipeline_task else True,
        "user":                   current_user.username,
        "total_frames_collected": sum(len(v.get("per_frame_details", v.get("frames_index", []))) for v in all_active.values()),
        "videos_in_pipeline":     len([v for v in all_active.values() if v.get("input_type") == "video"]),
        "images_analysed":        len([v for v in all_active.values() if v.get("input_type") == "images"]),
        "pdf_files_analysed":     len([v for v in all_active.values() if v.get("input_type") in ("pdf", "document")]),
        "audio_files_analysed":   len(audios),
        "stored_lectures_total": len(db_stems),
        "videos":                 video_states,
    })


# ──────────────────────────────────────────────────────────────────────────────
#  RESULTS  (auth added; DB fallback on every GET)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/results/video", summary="Full JSON results for all videos", tags=["Results"])
async def results_video(
    current_user: User = Depends(auth.get_current_user),
) -> List[Dict[str, Any]]:
    return [
        v for v in academic_results.values()
        if v.get("input_type") == "video" and v.get("user_id") == current_user.id
    ]


@app.get("/results/image", summary="Analysis results for all images", tags=["Results"])
async def results_image(
    current_user: User = Depends(auth.get_current_user),
) -> List[Dict[str, Any]]:
    return [
        v for v in academic_results.values()
        if v.get("input_type") == "images" and v.get("user_id") == current_user.id
    ]


@app.get("/results/audio", summary="Transcription results for all audio", tags=["Results"])
async def results_audio(
    current_user: User = Depends(auth.get_current_user),
) -> List[Dict[str, Any]]:
    combined = []
    mem_stems = set()
    
    # 1. In-memory
    for v in academic_results.values():
        if v.get("user_id") != current_user.id:
            continue
        if v.get("input_type") == "audio" or (v.get("input_type") == "video" and (v.get("audio_analysis") or v.get("transcription"))):
            src = v.get("audio_path") or v.get("video_path")
            trans = v.get("transcription") or v.get("audio_analysis")
            if trans:
                # Attempt to extract stem
                stem = v.get("video_stem") or v.get("batch_stem")
                if not stem and src:
                    stem = Path(src).stem
                if stem:
                    mem_stems.add(stem)
                combined.append({
                    **v,
                    "audio_path":    src,
                    "transcription": trans,
                    "from_video":    v.get("input_type") == "video",
                    "video_stem":    stem,
                    "lecture_title": v.get("lecture_summary", {}).get("lecture_title") or v.get("display_name") or stem
                })
                
    # 2. Add from Database
    from database_v2 import SessionLocal, Media, Video, TranscriptionSegment
    db = SessionLocal()
    try:
        db_media = db.query(Media).filter(
            Media.user_id == current_user.id,
            Media.media_type.in_(["video", "audio"])
        ).all()
        
        for m in db_media:
            stem = m.batch_stem or Path(m.storage_path or "").stem
            if stem in mem_stems:
                continue
                
            transcription = None
            lecture_title = None
            duration_sec = 0
            language = "en"
            
            # Try to get transcription from Video row first (legacy/summary)
            if m.media_type == "video":
                v_row = db.query(Video).filter(Video.media_id == m.id).first()
                if v_row:
                    transcription = v_row.transcription
                    lecture_title = v_row.lecture_title
                    duration_sec = v_row.duration_sec
                    language = v_row.detected_language or "en"
            
            # Get stats from the new summary table if available
            from database_v2 import MediaResultStats
            stats = db.query(MediaResultStats).filter(MediaResultStats.media_id == m.id).first()
            if stats:
                duration_sec = stats.duration_sec or duration_sec
                language = stats.language_code or language

            # Reconstruct granular segments if they exist in the new table
            db_segments = db.query(TranscriptionSegment).filter(
                TranscriptionSegment.media_id == m.id
            ).order_by(TranscriptionSegment.segment_index).all()
            
            if db_segments:
                seg_list = [
                    {
                        "id":         s.segment_index,
                        "start":      s.start_time,
                        "end":        s.end_time,
                        "text":       s.text,
                        "confidence": s.confidence
                    }
                    for s in db_segments
                ]
                full_text = " ".join([s["text"] for s in seg_list])
                transcription = {
                    "text":     full_text,
                    "segments": seg_list,
                    "backend":  "database",
                    "language": language
                }
                if not duration_sec and seg_list:
                    duration_sec = seg_list[-1]["end"]

            if transcription:
                combined.append({
                    "audio_path":    m.storage_path,
                    "video_path":    m.storage_path if m.media_type == "video" else None,
                    "video_stem":    stem,
                    "batch_stem":    m.batch_stem,
                    "transcription": transcription,
                    "from_video":    m.media_type == "video",
                    "lecture_title": lecture_title or stem,
                    "duration_sec":  duration_sec,
                    "detected_language": language,
                    "error":         None
                })
    finally:
        db.close()
        
    return combined


@app.get("/results/notes/{video_stem}", summary="Markdown study notes", tags=["Results"])
async def results_notes(
    video_stem: str,
    current_user: User = Depends(auth.get_current_user),
) -> PlainTextResponse:
    # 1. In-memory (fastest path)
    v = _find_any_lecture(video_stem, current_user.id)
    if v and v.get("user_id") == current_user.id:
        notes = v.get("study_notes")
        if notes:
            return PlainTextResponse(notes, media_type="text/markdown")
        err = v.get("error")
        if err:
            raise HTTPException(500, f"Pipeline error: {err}")
        raise HTTPException(503, "Study notes not yet generated — check /status.")

    # 2. Database only
    from database_v2 import SessionLocal, Media, Note
    db = SessionLocal()
    try:
        media = db.query(Media).filter(Media.batch_stem == video_stem, Media.user_id == current_user.id).first()
        if media and media.notes:
            return PlainTextResponse(media.notes[0].content, media_type="text/markdown")
    finally:
        db.close()

    raise HTTPException(404, f"No study notes found for '{video_stem}'. Generate them first.")


@app.get("/results/pdf/{video_stem}", summary="Download PDF academic report", tags=["Results"])
async def results_pdf(
    video_stem: str,
    current_user: User = Depends(auth.get_current_user),
) -> FileResponse:
    # 1. In-memory
    v = _find_any_lecture(video_stem, current_user.id)
    if v and v.get("user_id") == current_user.id:
        pdf_path = v.get("pdf_report_path")
        if pdf_path and os.path.isfile(pdf_path):
            return FileResponse(pdf_path, media_type="application/pdf",
                                filename=os.path.basename(pdf_path))
        err = v.get("error")
        if err:
            raise HTTPException(500, f"Pipeline error: {err}")
        raise HTTPException(503, "PDF not yet generated — check /status.")

    # 2. Database only
    from db_actions import get_db_media_states
    db_states = get_db_media_states(str(current_user.id))
    
    if video_stem in db_states:
        from database_v2 import SessionLocal, Media
        db = SessionLocal()
        try:
            media = db.query(Media).filter(Media.batch_stem == video_stem, Media.user_id == current_user.id).first()
            if media and media.pdf_report_path:
                if os.path.isfile(media.pdf_report_path):
                    return FileResponse(media.pdf_report_path, media_type="application/pdf",
                                        filename=os.path.basename(media.pdf_report_path))
        finally:
            db.close()

    raise HTTPException(404, f"No PDF found for '{video_stem}'. Generate it first.")


@app.get(
    "/results/flashcards/{video_stem}",
    summary="Q&A flashcards (requires POST /generate/flashcards/{stem} first)",
    tags=["Flashcards & Quiz"],
)
async def results_flashcards(
    video_stem: str,
    current_user: User = Depends(auth.get_current_user),
) -> List[Dict]:
    # 1. In-memory
    v = _find_any_lecture(video_stem, current_user.id)
    if v and v.get("user_id") == current_user.id:
        cards = v.get("flashcards")
        if cards:
            return cards
        # If not in memory, don't raise 404 here; fall through to DB or disk
        fc_state = _flashcard_states.get(video_stem, {}).get("state", "idle")
        if fc_state == "running":
            raise HTTPException(503, f"Still generating — poll GET /generate/flashcards/{video_stem}/status.")
        # Removed the idle -> 404 raise to support DB fallback for in-memory videos
    
    # 2. Database only
    from database_v2 import SessionLocal, Media, Flashcard
    db = SessionLocal()
    try:
        media = db.query(Media).filter(Media.batch_stem == video_stem, Media.user_id == current_user.id).first()
        if media and media.flashcards:
            return [
                {
                    "question":   c.question,
                    "answer":     c.answer,
                    "topic":      c.topic,
                    "difficulty": c.difficulty
                }
                for c in media.flashcards
            ]
    finally:
        db.close()

    raise HTTPException(404, f"No flashcards found. Call POST /generate/flashcards/{video_stem} first.")


@app.get(
    "/results/quiz/{video_stem}",
    summary="MCQ quiz (requires POST /generate/flashcards/{stem} first)",
    tags=["Flashcards & Quiz"],
)
async def results_quiz(
    video_stem: str,
    current_user: User = Depends(auth.get_current_user),
) -> List[Dict]:
    # 1. In-memory
    v = _find_any_lecture(video_stem, current_user.id)
    if v and v.get("user_id") == current_user.id:
        quiz = v.get("quiz")
        if quiz:
            return quiz
        # If not in memory, don't raise 404 here; fall through to DB or disk
        fc_state = _flashcard_states.get(video_stem, {}).get("state", "idle")
        if fc_state == "running":
            raise HTTPException(503, f"Still generating — poll GET /generate/flashcards/{video_stem}/status.")
        # Removed the idle -> 404 raise to support DB fallback for in-memory videos
    
    # 2. Database only
    from database_v2 import SessionLocal, Media, QuizQuestion
    db = SessionLocal()
    try:
        media = db.query(Media).filter(Media.batch_stem == video_stem, Media.user_id == current_user.id).first()
        if media and media.quiz_questions:
            return [
                {
                    "question":       q.question,
                    "options":        q.options,
                    "correct_answer": q.correct_answer,
                    "explanation":    q.explanation,
                    "topic":          q.topic
                }
                for q in media.quiz_questions
            ]
    finally:
        db.close()

    raise HTTPException(404, f"No quiz found. Call POST /generate/flashcards/{video_stem} first.")


@app.get("/results/graph/{video_stem}", summary="Knowledge graph in D3.js format", tags=["Results"])
async def results_graph(
    video_stem: str,
    current_user: User = Depends(auth.get_current_user),
) -> Dict:
    # 1. In-memory
    v = _find_any_lecture(video_stem, current_user.id)
    if v and v.get("user_id") == current_user.id:
        kg = v.get("knowledge_graph")
        if kg is not None:
            return kg
        err = v.get("error")
        if err:
            raise HTTPException(500, f"Pipeline error: {err}")
        raise HTTPException(503, "Knowledge graph not yet generated — check /status.")

    # 2. Database only
    from database_v2 import SessionLocal, Media
    db = SessionLocal()
    try:
        media = db.query(Media).filter(Media.batch_stem == video_stem, Media.user_id == current_user.id).first()
        if media and media.knowledge_graph:
            return json.loads(media.knowledge_graph) if isinstance(media.knowledge_graph, str) else media.knowledge_graph
    finally:
        db.close()

    raise HTTPException(404, f"No knowledge graph found for '{video_stem}'. Generate it first.")


@app.get("/results/frames/{video_stem}", summary="Frame index", tags=["Results"])
async def results_frames(
    video_stem: str,
    current_user: User = Depends(auth.get_current_user),
) -> List[Dict]:
    v = _find_any_lecture(video_stem, current_user.id)
    if v and v.get("user_id") == current_user.id:
        return v.get("frames_index", [])
    p = os.path.join(FRAMES_BASE_DIR, video_stem, "frames_index.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(404, f"No frame index found for '{video_stem}'.")


@app.get("/results/latest", summary="Most recent N frames", tags=["Results"])
async def latest(
    n: int = 10,
    current_user: User = Depends(auth.get_current_user),
) -> List[Dict]:
    all_frames = [
        {**fr, "video_path": v["video_path"]}
        for v in academic_results.values()
        if v.get("input_type") == "video" and v.get("user_id") == current_user.id
        for fr in v["per_frame_details"]
    ]
    return all_frames[-n:]


@app.get("/video/{video_stem}", summary="Stream uploaded video", tags=["Results"])
async def stream_video(
    video_stem: str,
    token: Optional[str] = Query(None),
) -> FileResponse:
    # Manual token validation for video streams (to allow browser <video> player)
    current_user = None
    if not token:
        raise HTTPException(401, "No authentication token provided in query.")
        
    try:
        current_user = auth.decode_access_token(token)
    except Exception:
        raise HTTPException(401, "Invalid or expired token.")

    if not current_user:
        raise HTTPException(401, "User not found.")

    video_path = None
    v = _find_any_lecture(video_stem, current_user.id)
    if v and v.get("user_id") == current_user.id:
        video_path = v.get("video_path")
        
        # Support for Live Lecture audio playback
        if v.get("input_type") == "live" and v.get("live_session_id"):
            session_id = v.get("live_session_id")
            audio_path = os.path.join("live_audio", session_id, "final_merged.wav")
            if os.path.isfile(audio_path):
                return FileResponse(audio_path, media_type="audio/wav")

    if not video_path:
        # Final fallback - try to find in uploads dir
        for f in os.listdir(UPLOAD_DIR):
            if f.startswith(f"{current_user.id}_") and Path(f).stem == video_stem:
                video_path = os.path.join(UPLOAD_DIR, f)
                break
                
    if not video_path:
        raise HTTPException(404, f"Video record not found for stem '{video_stem}'.")
        
    # Redirect to MinIO presigned URL
    try:
        from minio_utils import minio_client
        safe_name = os.path.basename(video_path)
        url = minio_client.get_presigned_url(f"videos/{safe_name}")
        return RedirectResponse(url)
    except Exception as e:
        logger.error(f"Failed to get presigned URL for video: {e}")
        # Fallback to local file if MinIO fails
        if os.path.isfile(video_path):
            return FileResponse(video_path)
        raise HTTPException(404, f"Video file not found and MinIO retrieval failed.")


@app.get("/media/images/{stem}/{filename}", summary="Serve image right from MinIO", tags=["Media"])
async def serve_minio_image(stem: str, filename: str):
    try:
        from minio_utils import minio_client
        from fastapi.responses import RedirectResponse
        url = minio_client.get_presigned_url(f"images/{stem}/{filename}")
        return RedirectResponse(url)
    except Exception as e:
        # Try without stem if it fails (backwards compat)
        try:
            url = minio_client.get_presigned_url(f"images/{filename}")
            return RedirectResponse(url)
        except:
            raise HTTPException(404, f"Failed to retrieve image from MinIO: {e}")

@app.get("/api/v1/probe", tags=["Control"])
async def probe_v1():
    return {"message": "Server is running v1 API"}


@app.delete("/api/v1/lecture/{video_stem}", summary="Delete a lecture and all its data permanently", tags=["Control"])
async def delete_lecture(
    video_stem: str,
    current_user: User = Depends(auth.get_current_user),
) -> JSONResponse:
    video_path = None
    audio_path = None
    v = _find_any_lecture(video_stem, current_user.id)
    if v and v.get("user_id") == current_user.id:
        video_path = v.get("video_path")
        audio_path = v.get("_audio_path")
    if not video_path:
        video_path = _find_uploaded_source(video_stem, _stem_input_type(video_stem))
    if not audio_path:
        audio_path = os.path.join(AUDIO_DIR, f"{video_stem}.wav")

    # 2. Delete local files
    try:
        # Video file
        if video_path and os.path.isfile(video_path):
            os.remove(video_path)
        
        # Audio file
        if audio_path and os.path.isfile(audio_path):
            os.remove(audio_path)
            
        # Extracted frames directory
        frames_dir = os.path.join(FRAMES_BASE_DIR, video_stem)
        if os.path.isdir(frames_dir):
            shutil.rmtree(frames_dir)
            
        # Study notes
        notes_path = os.path.join(NOTES_DIR, f"{video_stem}_study_notes.md")
        if os.path.isfile(notes_path): os.remove(notes_path)
            
        # PDF report
        pdf_path = os.path.join(PDF_DIR, f"{video_stem}_academic_report.pdf")
        if os.path.isfile(pdf_path): os.remove(pdf_path)
            
        # Quiz & Flashcards (json version)
        fc_path = os.path.join(FLASHCARD_DIR, f"{video_stem}_flashcards.json")
        if os.path.isfile(fc_path): os.remove(fc_path)
        qz_path = os.path.join(QUIZ_DIR, f"{video_stem}_quiz.json")
        if os.path.isfile(qz_path): os.remove(qz_path)
        
        # Knowledge graph
        kg_path = os.path.join(GRAPH_DIR, f"{video_stem}_knowledge_graph.json")
        if os.path.isfile(kg_path): os.remove(kg_path)
        
    except Exception as e:
        logger.warning(f"Failed to clean up some local files: {e}")

    # 3. Delete from MinIO
    try:
        prefixes = [f"videos/{video_stem}", f"audio/{video_stem}", f"images/{video_stem}"]
        for pref in prefixes:
            objects_to_delete = minio_client.client.list_objects(minio_client.MINIO_BUCKET_NAME, prefix=pref, recursive=True)
            for obj in objects_to_delete:
                minio_client.client.remove_object(minio_client.MINIO_BUCKET_NAME, obj.object_name)
    except Exception as e:
        logger.warning(f"Failed to clean up MinIO objects: {e}")

    # 4. Remove from DB so DB-backed library cards do not reappear after refresh
    try:
        from database_v2 import SessionLocal, Media

        db = SessionLocal()
        try:
            media_rows = (
                db.query(Media)
                .filter(Media.user_id == current_user.id, Media.batch_stem == video_stem)
                .all()
            )

            if not media_rows:
                media_rows = [
                    m for m in db.query(Media).filter(Media.user_id == current_user.id).all()
                    if Path(m.storage_path or "").stem == video_stem
                ]

            for media in media_rows:
                db.delete(media)

            if media_rows:
                db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Failed to delete DB lecture state for '{video_stem}': {e}")

    # 5. Remove from in-memory results and task state
    targets = [
        k for k, v in academic_results.items()
        if (
            Path(k).stem == video_stem
            or Path(str(v.get("video_path", ""))).stem == video_stem
            or v.get("video_stem") == video_stem
        )
    ]
    for tk in targets:
        if academic_results[tk].get("user_id") == current_user.id:
            del academic_results[tk]
    _flashcard_states.pop(video_stem, None)
    _flashcard_tasks.pop(video_stem, None)

    return JSONResponse({"message": f"Lecture '{video_stem}' and all associated data deleted successfully."})


# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
#  v3.2.1: DASHBOARD  (DB-backed)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/dashboard/stats", summary="Per-user lecture statistics from DB", tags=["Dashboard"])
def dashboard_stats(
    current_user: User = Depends(auth.get_current_user),
) -> JSONResponse:
    now_ts = time.time()
    last_48h_ts = now_ts - (48 * 60 * 60)
    from db_actions import get_db_media_states
    db_states_dict = get_db_media_states(str(current_user.id))
    disk_states = list(db_states_dict.values())
    
    # Include live lectures
    from database_v2 import SessionLocal, LiveLecture
    db = SessionLocal()
    try:
        live_lectures_db = db.query(LiveLecture).filter(LiveLecture.user_id == current_user.id).all()
        for ll in live_lectures_db:
            disk_states.append({
                "video_stem": ll.pipeline_stem or ll.session_id,
                "display_name": ll.title,
                "input_type": "live",
                "created_at": ll.created_at.timestamp() if ll.created_at else 0
            })
    except Exception as e:
        pass
    finally:
        db.close()

    total_lectures = len([s for s in disk_states if s["input_type"] == "video"])
    total_images = len([s for s in disk_states if s["input_type"] == "images"])
    total_docs = len([s for s in disk_states if s["input_type"] == "document"])
    total_audios = len([s for s in disk_states if s["input_type"] == "audio"])
    total_live = len([s for s in disk_states if s["input_type"] == "live"])

    last_48h_lectures = len([s for s in disk_states if s["input_type"] == "video" and s["created_at"] >= last_48h_ts])
    last_48h_images = len([s for s in disk_states if s["input_type"] == "images" and s["created_at"] >= last_48h_ts])
    last_48h_docs = len([s for s in disk_states if s["input_type"] == "document" and s["created_at"] >= last_48h_ts])
    last_48h_audios = len([s for s in disk_states if s["input_type"] == "audio" and s["created_at"] >= last_48h_ts])

    recent_lectures = sorted(
        disk_states,
        key=lambda item: item.get("created_at", 0),
        reverse=True,
    )[:10]

    from db_actions import get_db_user_progress_stats, get_study_recommendations
    progress_stats = get_db_user_progress_stats(str(current_user.id))
    study_recs = get_study_recommendations(str(current_user.id))

    # ── Subject area distribution for donut chart ──
    subject_counts = {}
    for s in disk_states:
        subj = (s.get("subject_area") or "").strip()
        if subj:
            # Collapse sub-topics into main subjects (e.g. "Physics - Kinematics" -> "Physics")
            # We look for common separators: " - ", ":", " > ", "-", "/", or "|"
            for sep in [" - ", ":", " > ", "-", "/", "|"]:
                if sep in subj:
                    subj = subj.split(sep)[0].strip()
                    break

            
            if subj:
                # Standardize capitalization (e.g., "physics" -> "Physics")
                subj = subj.title()
                subject_counts[subj] = subject_counts.get(subj, 0) + 1



    return JSONResponse({
        "total_lectures":  total_lectures,
        "total_images":    total_images,
        "total_audios":    total_audios,
        "total_docs":      total_docs,
        "last_48h": {
            "videos": last_48h_lectures,
            "images": last_48h_images,
            "audios": last_48h_audios,
            "docs":   last_48h_docs,
        },
        "total_live": total_live,
        "engagement": progress_stats,
        "subject_distribution": subject_counts,
        "study_recommendations": study_recs,
        "recent_lectures": [
            {
                "stem": item["video_stem"],
                "title": item["display_name"],
                "date": datetime.fromtimestamp(item["created_at"]).isoformat() if item["created_at"] else None,
                "type": item["input_type"],
            }
            for item in recent_lectures
        ],
    })


@app.post("/dashboard/generate-plan/{stem}", summary="Generate AI Study Plan", tags=["Dashboard"])
async def generate_study_plan_endpoint(
    stem: str,
    current_user: User = Depends(auth.get_current_user)
):
    """
    Generate a 1-paragraph personalized study plan based on lecture context.
    """
    global _shared_llm
    from db_actions import get_lecture_context_for_plan
    
    ctx = get_lecture_context_for_plan(str(current_user.id), stem)
    if not ctx:
        raise HTTPException(404, f"Lecture context for stem '{stem}' not found.")

    # Format context for LLM
    quiz_info = f"Recent quiz scores: {ctx['quiz_scores']}" if ctx['quiz_scores'] else "No quiz attempts yet."
    flash_info = f"Flashcards: {ctx['flashcard_stats']['reviewed']}/{ctx['flashcard_stats']['total']} reviewed (Avg Confidence: {ctx['flashcard_stats']['avg_confidence']:.1f}/5.0)"
    
    topics_str = ", ".join(ctx['topics']) if ctx['topics'] else "None listed"
    
    prompt_context = f"""
Lecture: {ctx['title']}
Subject: {ctx['subject']}
Topics: {topics_str}
Summary: {ctx['summary'][:500]}...
Performance: {quiz_info}. {flash_info}.
"""

    sys_prompt = """You are an expert academic tutor. Your goal is to provide a concise, high-impact 1-paragraph study plan (3-5 sentences) for a specific lecture based on the user's performance and the lecture content.
Focus on what they should do NEXT to improve (e.g., "Re-read the section on X", "Attempt another quiz", or "Review flashcards with low confidence").
Be encouraging but direct and academic. Do NOT use bullet points. DO NOT use greetings like "Hello" or "Sure"."""

    _shared_llm = get_or_load_shared_llm("Dashboard Plan")

    full_prompt = f"<|system|>\n{sys_prompt}<|end|>\n<|user|>\nProvide a study plan for:\n{prompt_context}<|end|>\n<|assistant|>\n"
    
    try:
        response_text = _llm_reason_text(_shared_llm, full_prompt, 250)
        response_text = response_text.replace("<|end|>", "").replace("<|endoftext|>", "").strip()
        return {"plan": response_text}
    except Exception as e:
        logger.error(f"[Dashboard Plan] failed: {e}")
        raise HTTPException(500, f"Error generating study plan: {e}")


@app.post("/dashboard/generate-mindmap/{stem}", summary="Generate Mermaid.js Mind Map", tags=["Dashboard"])
async def generate_mindmap_endpoint(
    stem: str,
    current_user: User = Depends(auth.get_current_user)
):
    """
    Generate a Mermaid.js mindmap string for the lecture structure.
    """
    global _shared_llm
    from db_actions import get_lecture_context_for_plan
    
    ctx = get_lecture_context_for_plan(str(current_user.id), stem)
    if not ctx:
        raise HTTPException(404, f"Lecture context for stem '{stem}' not found.")

    from academic_system.prompts1 import prompt_mindmap
    # We pass topics and concepts for a hierarchical structure
    p_text = prompt_mindmap(ctx['title'], ctx['subject'], ctx['topics'], ctx['concepts'])
    full_prompt = f"<|user|>\n{p_text}<|end|>\n<|assistant|>\n"

    _shared_llm = get_or_load_shared_llm("Dashboard Mindmap")

    try:
        response_text = _llm_reason_text(_shared_llm, full_prompt, 500)
        # Clean up tags
        response_text = response_text.replace("<|end|>", "").replace("<|endoftext|>", "").strip()
        
        # Strip markdown code fences if model included them
        if "```mermaid" in response_text:
            response_text = response_text.split("```mermaid")[1].split("```")[0].strip()
        elif "```" in response_text:
            # handle case where it just says ```
            parts = response_text.split("```")
            if len(parts) >= 3:
                response_text = parts[1].strip()
            else:
                response_text = parts[0].strip()
        
        # Ensure it starts with mindmap
        if not response_text.startswith("mindmap") and "mindmap" in response_text:
            response_text = "mindmap" + response_text.split("mindmap")[1]
            
        return {"mindmap": response_text}
    except Exception as e:
        logger.error(f"[Dashboard Mindmap] failed: {e}")
        raise HTTPException(500, f"Error generating mind map: {e}")



# ──────────────────────────────────────────────────────────────────────────────
#  STUDENT PROGRESS API  (auth added; unchanged otherwise)
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
    current_user: User = Depends(auth.get_current_user),
) -> JSONResponse:
    global _progress

    v = _find_any_lecture(video_stem, current_user.id)
    if v and v.get("user_id") == current_user.id:
        flashcards = v.get("flashcards", [])
        if card_index < 0 or card_index >= len(flashcards):
            raise HTTPException(
                404,
                f"Card index {card_index} out of range (0–{len(flashcards) - 1}).",
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
        "user_id":     current_user.id,
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
    session_id: Optional[str] = Query(None),
    current_user: User        = Depends(auth.get_current_user),
) -> Dict:
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
    current_user: User        = Depends(auth.get_current_user),
) -> List[Dict]:
    v          = _find_any_lecture(video_stem, current_user.id)
    flashcards = v.get("flashcards", []) if v else []

    if not flashcards:
        p = os.path.join(FLASHCARD_DIR, f"{video_stem}_flashcards.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                flashcards = json.load(f)

    if not flashcards:
        raise HTTPException(
            404,
            f"No flashcards found for '{video_stem}'. "
            f"Call POST /generate/flashcards/{video_stem} first.",
        )

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

    due.sort(
        key=lambda c: (
            0 if c["priority"] == "never_reviewed" else 1,
            c["last_reviewed"] or 0,
        )
    )
    return due[:limit]


@app.post(
    "/quiz/session/{video_stem}",
    summary="Record a completed quiz session summary",
    tags=["Student Progress"],
)
async def quiz_session(
    video_stem: str,
    body:       QuizSessionRequest,
    current_user: User = Depends(auth.get_current_user),
) -> JSONResponse:
    """Persist a quiz score summary to the database for accuracy tracking."""
    from db_actions import save_quiz_session_to_db
    
    save_quiz_session_to_db(
        user_id         = str(current_user.id),
        video_stem      = video_stem,
        total_questions = body.total_questions,
        correct_answers = body.correct_answers,
    )

    return JSONResponse({
        "message": "Quiz session recorded.",
        "accuracy": (body.correct_answers / body.total_questions * 100) if body.total_questions > 0 else 0
    })


# ──────────────────────────────────────────────────────────────────────────────
#  LANGUAGE INFO  (auth added)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/languages", summary="Supported OCR + transcription languages", tags=["Info"])
async def list_languages(
    current_user: User = Depends(auth.get_current_user),
) -> Dict:
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
#  HOME  (public)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Home"])
async def root():
    return {"message": "AcademIQ v3.2.1 — Multi-User Academic Pipeline is online."}


# ──────────────────────────────────────────────────────────────────────────────
#  DIAGNOSTICS  (auth added; unchanged content)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/diagnostics", summary="System health and endpoint reference", tags=["Health"])
async def diagnostics(
    current_user: User = Depends(auth.get_current_user),
) -> JSONResponse:
    from video_pipeline.detection.ocr    import EASYOCR_AVAILABLE, TESSERACT_AVAILABLE
    from academic_system.slide_detector  import SKIMAGE_AVAILABLE
    from academic_system.deduplicator    import ST_AVAILABLE, SKLEARN_AVAILABLE
    from academic_system.knowledge_graph import NX_AVAILABLE

    try:
        import certifi as _certifi
        certifi_available = True
        certifi_path      = _certifi.where()
    except ImportError:
        certifi_available = False
        certifi_path      = None

    # DB Health Check
    db_ok = False
    db_err = None
    try:
        from database_v2 import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db_ok = True
        db.close()
    except Exception as e:
        db_err = str(e)

    return JSONResponse({
        "system":    "Multi-Modal Academic Intelligence System v3.2.1",
        "database":  {"ok": db_ok, "error": db_err},
        "reasoning": config.reasoning_model_id,
        "auth":      "JWT Bearer (PostgreSQL-backed users)",
        "v321_additions": {
            "auth_endpoints":    ["POST /auth/register", "POST /auth/login", "GET /auth/me"],
            "db_persistence":    "Lectures, flashcards, quiz questions persisted to PostgreSQL",
            "user_isolation":    "Results filtered by authenticated user_id",
            "db_fallback":       "GET endpoints fall back to DB when result is not in memory",
            "dashboard":         "GET /dashboard/stats — per-user lecture history",
        },
        "v310_long_video_config": {
            "adaptive_fps_threshold_1_min": _FPS_LONG_THRESHOLD_1 // 60,
            "adaptive_fps_for_long_1":      _FPS_FOR_LONG_1,
            "adaptive_fps_threshold_2_min": _FPS_LONG_THRESHOLD_2 // 60,
            "adaptive_fps_for_long_2":      _FPS_FOR_LONG_2,
            "max_frames_extract":           _MAX_FRAMES_EXTRACT,
            "phase2_max_frames":            _PHASE2_MAX_FRAMES,
            "whisper_model":                _WHISPER_MODEL_SIZE,
            "whisper_timeout_sec":          _WHISPER_TIMEOUT_SEC,
            "ocr_batch_size":               int(os.environ.get("OCR_BATCH_SIZE", "8")),
            "ocr_workers":                  int(os.environ.get("OCR_WORKERS", "2")),
        },
        "phase2_strategy": {
            "mode":             "2-call split (notes only — no flashcards in pipeline)",
            "call1_budget":     _MAX_TOKENS_META,
            "call2_budget":     _MAX_TOKENS_NOTES,
            "flashcard_call":   "on-demand via POST /generate/flashcards/{stem}",
            "flashcard_budget": _MAX_TOKENS_CARDS,
        },
        "ssl_fix": {
            "certifi_available":      certifi_available,
            "certifi_ca_bundle":      certifi_path,
            "ssl_cert_file_env":      os.environ.get("SSL_CERT_FILE"),
            "requests_ca_bundle_env": os.environ.get("REQUESTS_CA_BUNDLE"),
        },
        "backends": {
            "easyocr":                 EASYOCR_AVAILABLE,
            "tesseract":               TESSERACT_AVAILABLE,
            "whisper_any":             WHISPER_AVAILABLE,
            "whisper_faster_whisper":  FASTER_WHISPER_AVAILABLE,
            "whisper_openai_fallback": OPENAI_WHISPER_AVAILABLE,
            "whisper_active_backend":  (
                "faster-whisper" if FASTER_WHISPER_AVAILABLE
                else "openai-whisper" if OPENAI_WHISPER_AVAILABLE
                else "none"
            ),
            "whisper_model":           _WHISPER_MODEL_SIZE,
            "skimage_ssim":            SKIMAGE_AVAILABLE,
            "sentence_transformers":   ST_AVAILABLE,
            "sklearn_tfidf":           SKLEARN_AVAILABLE,
            "networkx":                NX_AVAILABLE,
        },
        "pdf_pipeline": {
            "backends":          _check_pdf_backends(),
            "page_render_dpi":   int(os.environ.get("PDF_PAGE_DPI", "150")),
            "max_pages_phase2":  int(os.environ.get("PDF_MAX_PAGES_PHASE2", "40")),
            "install_hint":      "pip install pymupdf pdfplumber",
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
            "④ Flashcards   POST /generate/flashcards/{stem}  ← trigger",
            "             GET  /results/flashcards/{stem}     ← retrieve",
            "⑤ MCQ Quiz     GET /results/quiz/{stem}",
            "⑥ Graph (D3)   GET /results/graph/{stem}",
        ],
        "all_endpoints": {
            "register":             "POST   /auth/register",
            "login":                "POST   /auth/login",
            "me":                   "GET    /auth/me",
            "upload_video":         "POST   /upload/video",
            "upload_image":         "POST   /upload/image?language=en",
            "upload_audio":         "POST   /upload/audio",
            "upload_document":      "POST   /upload/document   (PDF full pipeline)",
            "status":               "GET    /status",
            "generate_flashcards":  "POST   /generate/flashcards/{stem}",
            "flashcard_gen_status": "GET    /generate/flashcards/{stem}/status",
            "results_json":         "GET    /results/video",
            "study_notes":          "GET    /results/notes/{stem}",
            "pdf":                  "GET    /results/pdf/{stem}",
            "flashcards":           "GET    /results/flashcards/{stem}",
            "quiz":                 "GET    /results/quiz/{stem}",
            "graph":                "GET    /results/graph/{stem}",
            "frames":               "GET    /results/frames/{stem}",
            "latest":               "GET    /results/latest?n=10",
            "review":               "PATCH  /flashcards/{stem}/{card_index}/review",
            "progress":             "GET    /progress/{stem}",
            "due_cards":            "GET    /progress/{stem}/due",
            "dashboard":            "GET    /dashboard/stats",
            "languages":            "GET    /languages",
            "stop":                 "POST   /stop",
            "clear":                "DELETE /results",
            "docs":                 "GET    /docs",
            "diagnostics":          "GET    /diagnostics",
        },
    })


# ──────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_v7-9:app", host="0.0.0.0", port=8000, reload=True)