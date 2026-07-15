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

# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, Depends, status
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

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
    WHISPER_AVAILABLE,
    FASTER_WHISPER_AVAILABLE,   # v3.2.0: faster-whisper backend flag
    OPENAI_WHISPER_AVAILABLE,   # v3.2.0: openai-whisper fallback flag
)

# ── v3 additions ──────────────────────────────────────────────────────────────
from academic_system.slide_detector   import SlideChangeDetector
from academic_system.deduplicator     import SemanticDeduplicator
from academic_system.knowledge_graph  import KnowledgeGraphBuilder
from academic_system.language_support import LanguageDetector

# ── v3.2.1: Auth & Database ──────────────────────────────────────────────────
from database_v2 import (
    init_db, get_db, SessionLocal,
    User, Lecture, Flashcard, QuizQuestion, StudentProgress,
    UploadedImage as Image, UploadedAudio as Audio,
    save_video_to_db, save_flashcards_to_db, save_image_to_db, save_audio_to_db,
)
import auth

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  v3.1.0: LONG-VIDEO CONSTANTS  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

_PHASE2_MAX_FRAMES: int = int(os.environ.get("PHASE2_MAX_FRAMES", "40"))

_FPS_LONG_THRESHOLD_1: int = int(os.environ.get("FPS_LONG_THRESHOLD_1", str(20 * 60)))
_FPS_LONG_THRESHOLD_2: int = int(os.environ.get("FPS_LONG_THRESHOLD_2", str(40 * 60)))
_FPS_FOR_LONG_1: float     = float(os.environ.get("FPS_FOR_LONG_1", "0.2"))
_FPS_FOR_LONG_2: float     = float(os.environ.get("FPS_FOR_LONG_2", "0.1"))

_MAX_FRAMES_EXTRACT: int = int(os.environ.get("MAX_FRAMES_EXTRACT", "720"))

_WHISPER_TIMEOUT_SEC: int = int(os.environ.get("WHISPER_TIMEOUT_SEC", "600"))

_WHISPER_MODEL_SIZE: str = os.environ.get("WHISPER_MODEL_SIZE", "tiny")


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


# ──────────────────────────────────────────────────────────────────────────────
#  GLOBAL STATE  (unchanged — academic_results keyed by video_path string)
# ──────────────────────────────────────────────────────────────────────────────

pipeline_task:    Optional[asyncio.Task] = None
pipeline_running: bool = False

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

# v3.1.0: shared OCR executor
_ocr_executor: Optional[Any] = None


# ──────────────────────────────────────────────────────────────────────────────
#  v3.2.1: DB STARTUP
# ──────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    try:
        init_db()
        logger.info("[STARTUP] PostgreSQL database initialised.")
    except Exception as exc:
        logger.error(f"[STARTUP] DB init error: {exc}")


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
    return (
        f"http://{SERVER_HOST}:{SERVER_PORT}"
        f"{SERVER_BASE_PATH}/{rel.replace(os.sep, '/')}"
    )


def _video_frames_dir(video_path: str) -> str:
    d = os.path.join(FRAMES_BASE_DIR, Path(video_path).stem)
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
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
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


# ── LLM helpers (unchanged) ───────────────────────────────────────────────────

_PHI3_CTX   = int(os.environ.get("PHI3_CONTEXT_LENGTH", "4096"))
_CTX_MARGIN = 64

_MAX_TOKENS_META  = int(os.environ.get("PHI3_MAX_TOKENS_META",  "300"))
_MAX_TOKENS_NOTES = int(os.environ.get("PHI3_MAX_TOKENS_NOTES", "700"))
_MAX_TOKENS_CARDS = int(os.environ.get("PHI3_MAX_TOKENS_CARDS", "1100"))


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
        logger.debug(
            f"[main] Token budget pre-capped: desired={desired} "
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


def _find_video(video_stem: str) -> Optional[Dict[str, Any]]:
    for v in academic_results.values():
        if v.get("input_type") == "video" and Path(v["video_path"]).stem == video_stem:
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


# ──────────────────────────────────────────────────────────────────────────────
#  IMAGE ANALYSIS  (unchanged)
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
#  ON-DEMAND FLASHCARD GENERATION  (v3.0.3, unchanged)
# ──────────────────────────────────────────────────────────────────────────────

async def _run_flashcard_generation_with_llm(video_stem: str, user_id: Optional[int] = None) -> None:
    """Reuses the pipeline's already-loaded LLM if available."""
    global _shared_llm
    state = _flashcard_states[video_stem]

    if _shared_llm is not None:
        logger.info(f"[Flashcards/{video_stem}] Reusing pipeline LLM — no reload needed.")
        await _run_flashcard_generation(video_stem, _shared_llm, user_id=user_id)
        return

    logger.info(f"[Flashcards/{video_stem}] No shared LLM found — loading fresh instance.")
    device = setup_device()
    try:
        llm = LlamaReasoner(
            model_id       = config.reasoning_model_id,
            max_new_tokens = config.max_reasoning_tokens,
            device         = device,
            load_in_4bit   = config.phi3_load_in_4bit,
            adapter_path   = config.phi3_adapter_path or None,
        )
        _shared_llm = llm
    except Exception as exc:
        logger.error(f"[Flashcards/{video_stem}] LLM load FAILED: {exc}", exc_info=True)
        state["state"] = "failed"
        state["error"] = f"LLM load failed: {exc}"
        return

    await _run_flashcard_generation(video_stem, llm, user_id=user_id)


async def _run_flashcard_generation(
    video_stem: str,
    llm: LlamaReasoner,
    user_id: Optional[int] = None,         # v3.2.1: used for DB persist
) -> None:
    """Background task — generates flashcards + quiz for a given video stem."""
    state = _flashcard_states[video_stem]
    state["state"] = "running"
    logger.info(f"[Flashcards/{video_stem}] Generation started.")

    try:
        vr = _find_video(video_stem)
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

        raw_text = _llm_reason_text(llm, prompt, _MAX_TOKENS_CARDS)
        llm._last_raw_output = raw_text
        raw = {}
        if raw_text:
            try:
                clean = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
                clean = re.sub(r"\s*```\s*$", "", clean.strip())
                parsed = json.loads(clean)
                if isinstance(parsed, dict):
                    raw = parsed
            except json.JSONDecodeError:
                pass

        if not raw:
            raw_text_for_recovery = getattr(llm, "_last_raw_output", "") or ""
            if raw_text_for_recovery:
                logger.warning(
                    f"[Flashcards/{video_stem}] Primary parse failed. "
                    f"Raw ({len(raw_text_for_recovery)} chars): {raw_text_for_recovery[:200]!r}"
                )
                clean_recovery = re.sub(r"^```(?:json)?\s*", "", raw_text_for_recovery.strip(), flags=re.IGNORECASE)
                clean_recovery = re.sub(r"\s*```\s*$", "", clean_recovery.strip())
                raw = _extract_first_json_object(clean_recovery) or {}
                if raw:
                    logger.info(f"[Flashcards/{video_stem}] JSON recovery (full obj): {list(raw.keys())}")
                else:
                    partial_cards = _partial_json_list(clean_recovery)
                    if partial_cards:
                        logger.info(
                            f"[Flashcards/{video_stem}] Partial recovery: "
                            f"{len(partial_cards)} item(s) from truncated output."
                        )
                        raw = {"flashcards": partial_cards, "quiz": []}

        flashcards: List[Dict] = []
        for card in raw.get("flashcards", []):
            if isinstance(card, dict) and card.get("question"):
                flashcards.append({
                    "question":   card.get("question", ""),
                    "answer":     card.get("answer", ""),
                    "topic":      card.get("topic", ""),
                    "difficulty": card.get("difficulty", "medium"),
                })

        quiz: List[Dict] = []
        for q in raw.get("quiz", []):
            if isinstance(q, dict) and q.get("question"):
                opts = {k: q[k] for k in ("A", "B", "C", "D") if k in q}
                if not opts and "options" in q:
                    opts = q["options"]
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

        video_path = vr["video_path"]

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

        # ── v3.2.1: persist flashcards and quiz to DB if context available ────
        if user_id is not None:
            try:
                # Find lecture_id from stem
                db_local = SessionLocal()
                lec_obj = db_local.query(Lecture).filter_by(lecture_stem=video_stem).first()
                if lec_obj:
                    save_flashcards_to_db(user_id, lec_obj.id, flashcards, quiz)
                db_local.close()
            except Exception as exc:
                logger.warning(f"[Flashcards/{video_stem}] DB persist failed (non-fatal): {exc}")

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
    user_id: Optional[int] = None,          # v3.2.1: for DB persist after Phase 2
) -> None:
    """
    Full v3.2.0 pipeline.  user_id is the only addition — it is passed
    straight through to save_lecture_to_db after Phase 2 completes.
    Every other line is identical to v3.2.0.
    """
    global pipeline_running, academic_results, stream_frame_counters, _shared_llm

    pipeline_running      = True
    stream_frame_counters = {}

    device = setup_device()

    effective_fps = config.fps
    if video_fps_overrides:
        lowest_fps = min(video_fps_overrides.values())
        if lowest_fps < config.fps:
            effective_fps = lowest_fps
            logger.info(
                f"[Pipeline v3.1.0] Adaptive FPS: using {effective_fps:.2f} fps "
                f"(config default is {config.fps} fps)."
            )

    _original_fps = config.fps
    config.fps    = effective_fps

    sources = {f"stream_{i}": p for i, p in enumerate(video_paths)}

    import concurrent.futures
    import threading

    stream_manager = StreamManager(sources, target_fps=effective_fps)
    config.fps = _original_fps

    _ocr_gpu_env = os.environ.get("OCR_USE_GPU")
    if _ocr_gpu_env is not None:
        ocr_use_gpu = device == "cuda" and _ocr_gpu_env.lower() == "true"
    else:
        ocr_use_gpu = device == "cuda" and config.easyocr_gpu

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

            if ocr_extractor is None:
                raise RuntimeError("OCRExtractor returned None.")
            if not callable(getattr(ocr_extractor, "extract", None)):
                raise RuntimeError("OCRExtractor does not expose .extract().")

            logger.info("Models ready ✓ (Phi-3 + EasyOCR)")

            global _shared_llm
            _shared_llm = llm

        except Exception as exc:
            _model_error[0] = exc
            logger.error(f"Model loading FAILED: {exc}", exc_info=True)
        finally:
            _models_ready.set()

    _model_thread = threading.Thread(target=_load_models, daemon=True, name="model_loader")
    _model_thread.start()

    _save_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="jpeg_save"
    )
    _executor = concurrent.futures.ThreadPoolExecutor(
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

    whisper_futures: Dict[str, concurrent.futures.Future] = {
        vp: _executor.submit(_run_whisper, vp)
        for vp in video_paths
    }
    logger.info(
        f"Started concurrently: frame extraction + model loading + "
        f"Whisper ({_WHISPER_MODEL_SIZE})."
    )

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
        if _pending_ocr:
            await _flush_ocr_batch_async(_pending_ocr)
            _pending_ocr.clear()

        stream_manager.release_all()
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
        whisper_results: Dict[str, Dict] = {}
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
            pipeline_running = False
            return

        # ── PHASE 2: per-video outputs ────────────────────────────────────────
        for stream_id, frames_list in per_stream_frames.items():
            video_path = stream_id_to_path[stream_id]
            vr         = academic_results[video_path]
            stem       = stem_path(video_path)

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
            if user_id is not None:
                try:
                    save_video_to_db(user_id, stem, vr)
                    logger.info(f"[DB] Lecture '{stem}' persisted for user {user_id}.")
                except Exception as exc:
                    logger.warning(f"[DB] save_video_to_db failed (non-fatal): {exc}")

            logger.info(
                f"[{stem}] Pipeline complete. "
                f"Call POST /generate/flashcards/{stem} when ready."
            )

        pipeline_running = False
        logger.info(
            "Academic pipeline v3.2.1 complete. "
            "Notes and PDF are ready. "
            "Use POST /generate/flashcards/{stem} to generate flashcards and quiz."
        )


# ──────────────────────────────────────────────────────────────────────────────
#  PYDANTIC MODELS
# ──────────────────────────────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    confidence: int           = Field(..., ge=1, le=5)
    correct:    bool          = Field(False)
    session_id: Optional[str] = Field(None)


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
async def register(body: UserRegister, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == body.username).first()
    if existing:
        raise HTTPException(400, "Username already registered.")
    new_user = User(
        username        = body.username,
        email           = body.email,
        hashed_password = auth.get_password_hash(body.password),
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    access_token = auth.create_access_token(data={"sub": new_user.username})
    return {
        "access_token": access_token,
        "token_type":   "bearer",
        "user":         {"id": new_user.id, "username": new_user.username},
    }


@app.post("/auth/login", tags=["Auth"], summary="Login and receive a JWT token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            401,
            "Incorrect username or password",
            {"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(data={"sub": user.username})
    return {
        "access_token": access_token,
        "token_type":   "bearer",
        "user":         {"id": user.id, "username": user.username},
    }


@app.get("/auth/me", tags=["Auth"], summary="Return the currently authenticated user")
async def me(current_user: User = Depends(auth.get_current_user)):
    return {"id": current_user.id, "username": current_user.username, "email": current_user.email}


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
        pipeline_running = False
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
        logger.info(f"Video saved → {dest}  (user={current_user.username})")

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
        }
        video_paths.append(dest)

    pipeline_task = asyncio.create_task(
        run_academic_pipeline(
            video_paths,
            video_fps_overrides=video_fps_overrides,
            user_id=current_user.id,            # v3.2.1
        )
    )

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
    current_user: User   = Depends(auth.get_current_user),   # v3.2.1
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
        safe_name = f"{current_user.id}_{upload.filename}"
        dest = os.path.join(IMAGE_DIR, safe_name)
        with open(dest, "wb") as fh:
            shutil.copyfileobj(upload.file, fh)

        try:
            analysis = process_image_academic(dest, ocr, llm, lang_info)
            entry    = {
                "input_type": "image",
                "image_path": dest,
                "language":   lang_info["name"],
                "analysis":   analysis,
                "user_id":    current_user.id,
            }
        except Exception as exc:
            logger.error(f"Image processing failed: {exc}")
            entry = {"input_type": "image", "image_path": dest, "error": str(exc)}

        academic_results[dest] = entry
        results.append(entry)
        if "analysis" in entry:
            save_image_to_db(current_user.id, dest, entry["analysis"])

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
    current_user: User = Depends(auth.get_current_user),     # v3.2.1
) -> JSONResponse:
    uploaded = [f for f in (file1, file2, file3) if f is not None]
    if not uploaded:
        raise HTTPException(400, "At least one audio file is required.")
    for f in uploaded:
        _assert_ext(f.filename, AUDIO_EXTENSIONS, "audio")

    results = []
    for upload in uploaded:
        safe_name = f"{current_user.id}_{upload.filename}"
        dest = os.path.join(AUDIO_DIR, safe_name)
        with open(dest, "wb") as fh:
            shutil.copyfileobj(upload.file, fh)

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
        save_audio_to_db(current_user.id, dest, transcription)

    return JSONResponse({
        "message":           f"{len(results)} audio file(s) transcribed.",
        "results":           results,
        "whisper_available": WHISPER_AVAILABLE,
        "whisper_model":     _WHISPER_MODEL_SIZE,
        "note":              "Language is auto-detected by Whisper.",
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
    current_user: User    = Depends(auth.get_current_user),  # v3.2.1
    db: Session           = Depends(get_db),                 # v3.2.1
) -> JSONResponse:
    vr = _find_video(video_stem)
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
    global pipeline_running
    pipeline_running = False
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
async def status(
    current_user: User  = Depends(auth.get_current_user),   # v3.2.1
    db: Session         = Depends(get_db),                   # v3.2.1
) -> JSONResponse:
    # In-memory results filtered to current user
    videos = {
        p: v for p, v in academic_results.items()
        if v.get("input_type") == "video" and v.get("user_id") == current_user.id
    }
    images = {
        p: v for p, v in academic_results.items()
        if v.get("input_type") == "image" and v.get("user_id") == current_user.id
    }
    audios = {
        p: v for p, v in academic_results.items()
        if v.get("input_type") == "audio" and v.get("user_id") == current_user.id
    }

    # v3.2.1: also include lectures stored in DB that are no longer in memory
    db_lectures = db.query(Lecture).filter(Lecture.user_id == current_user.id).all()
    db_stems    = {l.video_stem for l in db_lectures}
    mem_stems   = {Path(p).stem for p in videos}

    video_states = {}
    for p, v in videos.items():
        ss    = v.get("slide_change_stats", {})
        lang  = v.get("detected_language") or {}
        stem  = Path(p).stem
        fc_st = _flashcard_states.get(stem, {})
        dur   = v.get("duration_sec")
        v_state = {
            "source":                   "in_memory",
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
            "flashcards_ready":         fc_st.get("state") == "done",
            "flashcard_count":          fc_st.get("flashcard_count", 0),
            "quiz_ready":               fc_st.get("state") == "done",
            "quiz_count":               fc_st.get("quiz_count", 0),
            "graph_ready":              v.get("knowledge_graph") is not None,
            "pdf_ready":                v.get("pdf_report_path") is not None,
            "lecture_title":            v.get("lecture_summary", {}).get("lecture_title"),
            "subject_area":             v.get("lecture_summary", {}).get("subject_area"),
            "difficulty":               v.get("lecture_summary", {}).get("difficulty_level"),
            "pipeline_error":           v.get("error"),
            "video_stem":               stem,
            "display_name":             stem.split('_', 1)[-1] if '_' in stem else stem,
            "flashcard_generate_url":   f"POST /generate/flashcards/{stem}",
        }

        # v3.2.1: if idle in memory, check if previously saved in DB
        if v_state["flashcards_generation_state"] == "idle":
             db_lec = next((l for l in db_lectures if l.video_stem == stem), None)
             if db_lec:
                 db_fc = db.query(Flashcard).filter(Flashcard.lecture_id == db_lec.id).count()
                 db_qc = db.query(QuizQuestion).filter(QuizQuestion.lecture_id == db_lec.id).count()
                 if db_fc > 0:
                     v_state["flashcards_ready"] = True
                     v_state["flashcard_count"] = db_fc
                 if db_qc > 0:
                     v_state["quiz_ready"] = True
                     v_state["quiz_count"] = db_qc

        video_states[os.path.basename(p)] = v_state

    # DB-only lectures (not in memory — e.g. from a previous server session)
    for lec in db_lectures:
        if lec.video_stem not in mem_stems:
            fc_count = db.query(Flashcard).filter(Flashcard.lecture_id == lec.id).count()
            qc_count = db.query(QuizQuestion).filter(QuizQuestion.lecture_id == lec.id).count()
            video_states[f"{lec.video_stem} (db)"] = {
                "source":              "database",
                "lecture_title":       lec.title,
                "study_notes_ready":   bool(lec.study_notes),
                "pdf_ready":           bool(lec.pdf_report and lec.pdf_report.local_path and os.path.exists(lec.pdf_report.local_path or "")),
                "flashcard_count":     fc_count,
                "flashcards_ready":    fc_count > 0,
                "quiz_count":          qc_count,
                "quiz_ready":           qc_count > 0,
                "graph_ready":         bool(lec.knowledge_graph),
                "video_stem":          lec.lecture_stem,
                "display_name":        lec.lecture_stem.split('_', 1)[-1] if '_' in lec.lecture_stem else lec.lecture_stem,
                "flashcard_generate_url": f"POST /generate/flashcards/{lec.lecture_stem}",
            }

    return JSONResponse({
        "pipeline_running":       pipeline_running,
        "task_done":              pipeline_task.done() if pipeline_task else True,
        "user":                   current_user.username,
        "total_frames_collected": sum(len(v["per_frame_details"]) for v in videos.values()),
        "videos_in_pipeline":     len(videos),
        "images_analysed":        len(images),
        "audio_files_analysed":   len(audios),
        "db_lectures_total":      len(db_lectures),
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
        if v.get("input_type") == "image" and v.get("user_id") == current_user.id
    ]


@app.get("/results/audio", summary="Transcription results for all audio", tags=["Results"])
async def results_audio(
    current_user: User = Depends(auth.get_current_user),
) -> List[Dict[str, Any]]:
    combined = []
    for v in academic_results.values():
        if v.get("user_id") != current_user.id:
            continue
        if v.get("input_type") == "audio" or (v.get("input_type") == "video" and v.get("audio_analysis")):
            src = v.get("audio_path") or v.get("video_path")
            trans = v.get("transcription") or v.get("audio_analysis")
            if trans:
                combined.append({
                    **v,
                    "audio_path":    src,
                    "transcription": trans,
                    "from_video":    v.get("input_type") == "video"
                })
    return combined


@app.get("/results/notes/{video_stem}", summary="Markdown study notes", tags=["Results"])
async def results_notes(
    video_stem: str,
    current_user: User = Depends(auth.get_current_user),
    db: Session        = Depends(get_db),
) -> PlainTextResponse:
    # 1. In-memory (fastest path)
    v = _find_video(video_stem)
    if v and v.get("user_id") == current_user.id:
        notes = v.get("study_notes")
        if notes:
            return PlainTextResponse(notes, media_type="text/markdown")
        err = v.get("error")
        if err:
            raise HTTPException(500, f"Pipeline error: {err}")
        raise HTTPException(503, "Study notes not yet generated — check /status.")

    # 2. DB fallback (v3.2.1)
    lec = db.query(Lecture).filter(
        Lecture.lecture_stem == video_stem,
        Lecture.user_id   == current_user.id,
    ).first()
    if lec and lec.study_notes:
        return PlainTextResponse(lec.study_notes.notes_md, media_type="text/markdown")

    # 3. Disk fallback (unauthenticated path preserved for backwards compat)
    p = os.path.join(NOTES_DIR, f"{video_stem}_study_notes.md")
    if os.path.isfile(p):
        return PlainTextResponse(open(p, encoding="utf-8").read(), media_type="text/markdown")

    raise HTTPException(404, f"No study notes found for '{video_stem}'.")


@app.get("/results/pdf/{video_stem}", summary="Download PDF academic report", tags=["Results"])
async def results_pdf(
    video_stem: str,
    current_user: User = Depends(auth.get_current_user),
    db: Session        = Depends(get_db),
) -> FileResponse:
    # 1. In-memory
    v = _find_video(video_stem)
    if v and v.get("user_id") == current_user.id:
        pdf_path = v.get("pdf_report_path")
        if pdf_path and os.path.isfile(pdf_path):
            return FileResponse(pdf_path, media_type="application/pdf",
                                filename=os.path.basename(pdf_path))
        err = v.get("error")
        if err:
            raise HTTPException(500, f"Pipeline error: {err}")
        raise HTTPException(503, "PDF not yet generated — check /status.")

    # 2. DB fallback (v3.2.1)
    lec = db.query(Lecture).filter(
        Lecture.lecture_stem == video_stem,
        Lecture.user_id      == current_user.id,
    ).first()
    if lec and lec.pdf_report and lec.pdf_report.local_path and os.path.isfile(lec.pdf_report.local_path):
        return FileResponse(lec.pdf_report.local_path, media_type="application/pdf",
                            filename=f"{video_stem}.pdf")

    # 3. Disk fallback
    p = os.path.join(PDF_DIR, f"{video_stem}_academic_report.pdf")
    if os.path.isfile(p):
        return FileResponse(p, media_type="application/pdf", filename=os.path.basename(p))

    raise HTTPException(404, f"No PDF found for '{video_stem}'.")


@app.get(
    "/results/flashcards/{video_stem}",
    summary="Q&A flashcards (requires POST /generate/flashcards/{stem} first)",
    tags=["Flashcards & Quiz"],
)
async def results_flashcards(
    video_stem: str,
    current_user: User = Depends(auth.get_current_user),
    db: Session        = Depends(get_db),
) -> List[Dict]:
    # 1. In-memory
    v = _find_video(video_stem)
    if v and v.get("user_id") == current_user.id:
        cards = v.get("flashcards")
        if cards:
            return cards
        # If not in memory, don't raise 404 here; fall through to DB or disk
        fc_state = _flashcard_states.get(video_stem, {}).get("state", "idle")
        if fc_state == "running":
            raise HTTPException(503, f"Still generating — poll GET /generate/flashcards/{video_stem}/status.")
        # Removed the idle -> 404 raise to support DB fallback for in-memory videos
    
    # 2. DB fallback (v3.2.1)
    lec = db.query(Lecture).filter(
        Lecture.video_stem == video_stem,
        Lecture.user_id   == current_user.id,
    ).first()
    if lec:
        db_cards = db.query(Flashcard).filter(Flashcard.lecture_id == lec.id, Flashcard.user_id == current_user.id).all()
        if db_cards:
            return [{"question": c.question, "answer": c.answer,
                     "topic": getattr(c, "topic", ""), "difficulty": getattr(c, "difficulty", "medium")}
                    for c in db_cards]

    # 3. Disk fallback
    p = os.path.join(FLASHCARD_DIR, f"{video_stem}_flashcards.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            cards = json.load(f)
        if cards:
            return cards

    raise HTTPException(404, f"No flashcards found. Call POST /generate/flashcards/{video_stem} first.")


@app.get(
    "/results/quiz/{video_stem}",
    summary="MCQ quiz (requires POST /generate/flashcards/{stem} first)",
    tags=["Flashcards & Quiz"],
)
async def results_quiz(
    video_stem: str,
    current_user: User = Depends(auth.get_current_user),
    db: Session        = Depends(get_db),
) -> List[Dict]:
    # 1. In-memory
    v = _find_video(video_stem)
    if v and v.get("user_id") == current_user.id:
        quiz = v.get("quiz")
        if quiz:
            return quiz
        # If not in memory, don't raise 404 here; fall through to DB or disk
        fc_state = _flashcard_states.get(video_stem, {}).get("state", "idle")
        if fc_state == "running":
            raise HTTPException(503, f"Still generating — poll GET /generate/flashcards/{video_stem}/status.")
        # Removed the idle -> 404 raise to support DB fallback for in-memory videos
    
    # 2. DB fallback (v3.2.1) — quiz questions table
    lec = db.query(Lecture).filter(
        Lecture.video_stem == video_stem,
        Lecture.user_id   == current_user.id,
    ).first()
    if lec:
        db_quiz = db.query(QuizQuestion).filter(QuizQuestion.lecture_id == lec.id, QuizQuestion.user_id == current_user.id).all()
        if db_quiz:
            return [{"question": q.question, "options": q.options,
                     "correct_answer": q.correct_answer, "explanation": getattr(q, "explanation", "")}
                    for q in db_quiz]

    # 3. Disk fallback
    p = os.path.join(QUIZ_DIR, f"{video_stem}_quiz.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            quiz = json.load(f)
        if quiz:
            return quiz

    raise HTTPException(404, f"No quiz found. Call POST /generate/flashcards/{video_stem} first.")


@app.get("/results/graph/{video_stem}", summary="Knowledge graph in D3.js format", tags=["Results"])
async def results_graph(
    video_stem: str,
    current_user: User = Depends(auth.get_current_user),
    db: Session        = Depends(get_db),
) -> Dict:
    # 1. In-memory
    v = _find_video(video_stem)
    if v and v.get("user_id") == current_user.id:
        kg = v.get("knowledge_graph")
        if kg is not None:
            return kg
        err = v.get("error")
        if err:
            raise HTTPException(500, f"Pipeline error: {err}")
        raise HTTPException(503, "Knowledge graph not yet generated — check /status.")

    # 2. DB fallback (v3.2.1)
    lec = db.query(Lecture).filter(
        Lecture.video_stem == video_stem,
        Lecture.user_id   == current_user.id,
    ).first()
    if lec and lec.knowledge_graph:
        return lec.knowledge_graph

    # 3. Disk fallback
    p = os.path.join(GRAPH_DIR, f"{video_stem}_knowledge_graph.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    raise HTTPException(404, f"No knowledge graph found for '{video_stem}'.")


@app.get("/results/frames/{video_stem}", summary="Frame index", tags=["Results"])
async def results_frames(
    video_stem: str,
    current_user: User = Depends(auth.get_current_user),
) -> List[Dict]:
    v = _find_video(video_stem)
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


# ──────────────────────────────────────────────────────────────────────────────
#  v3.2.1: DASHBOARD  (DB-backed)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/dashboard/stats", summary="Per-user lecture statistics from DB", tags=["Dashboard"])
async def dashboard_stats(
    current_user: User = Depends(auth.get_current_user),
    db: Session        = Depends(get_db),
) -> JSONResponse:
    from datetime import datetime, timedelta
    last_48h_time = datetime.utcnow() - timedelta(hours=48)

    # Total counts
    total_lectures = db.query(Lecture).filter(Lecture.user_id == current_user.id).count()
    total_images   = db.query(Image).filter(Image.user_id == current_user.id).count()
    total_audios   = db.query(Audio).filter(Audio.user_id == current_user.id).count()
    total_docs     = db.query(Lecture).filter(
        Lecture.user_id == current_user.id,
        Lecture.pdf_report != None,
    ).count()

    # 48h counts
    last_48h_lectures = db.query(Lecture).filter(Lecture.user_id == current_user.id, Lecture.created_at >= last_48h_time).count()
    last_48h_images   = db.query(Image).filter(Image.user_id == current_user.id, Image.created_at >= last_48h_time).count()
    last_48h_audios   = db.query(Audio).filter(Audio.user_id == current_user.id, Audio.created_at >= last_48h_time).count()
    last_48h_docs     = db.query(Lecture).filter(
        Lecture.user_id == current_user.id,
        Lecture.pdf_report != None,
        Lecture.created_at >= last_48h_time,
    ).count()
    
    recent_lectures = db.query(Lecture).filter(Lecture.user_id == current_user.id).order_by(Lecture.created_at.desc()).limit(10).all()
    
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
        "recent_lectures": [
            {"stem": l.lecture_stem, "title": l.display_name, "date": str(l.created_at)}
            for l in recent_lectures
        ],
    })


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

    v = _find_video(video_stem)
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
    v          = _find_video(video_stem)
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

    return JSONResponse({
        "system":    "Multi-Modal Academic Intelligence System v3.2.1",
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