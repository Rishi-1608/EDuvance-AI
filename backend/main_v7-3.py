"""
Multi-Modal Academic Intelligence System  v3.2.0 (main_v7-2.py)
================================================
Transforms lecture videos, slide images, and audio recordings into
structured student learning materials.

What's new in v3.2.0 (faster-whisper + chunked transcription)
--------------------------------------------------------------
  ① faster-whisper primary backend — CTranslate2 INT8 GPU inference.
     4× faster than openai-whisper. Uses ~300 MB VRAM for tiny INT8
     vs ~500 MB for openai-whisper tiny (PyTorch overhead).
     Install: pip install faster-whisper

  ② Segment-level streaming — faster-whisper yields segments as a
     generator; the pipeline receives partial results as they complete
     rather than blocking until the full 1-hour audio finishes.

  ③ VAD filter — silence is skipped automatically; prevents the
     hallucination/hang on long pauses that stalled openai-whisper.

  ④ Per-segment + total timeouts — _CHUNK_TIMEOUT_SEC (default 60s)
     and _TOTAL_TIMEOUT_SEC (default 600s) ensure no single segment
     or the overall transcription can hang the pipeline.

  ⑤ Double-timeout fix in main — outer future.result() timeout is now
     _WHISPER_TIMEOUT_SEC + 120s so it never fires during normal
     faster-whisper partial-result collection.

  ⑥ Diagnostics updated — GET /diagnostics now shows active backend
     (faster-whisper vs openai-whisper) and both availability flags.

  ⑦ openai-whisper fallback — if faster-whisper is not installed,
     falls back to openai-whisper with explicit WAV chunking
     (_CHUNK_DURATION_SEC pieces, per-chunk timeout). Never blocks on
     a full 1-hour audio in a single call.

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
    The pipeline now ends after Call 2 (study notes + PDF).  This makes
    uploads faster and avoids the 45-second timeout that caused Call 3 to
    produce zero cards on RTX 3050.

  ② New endpoint:  POST /generate/flashcards/{video_stem}
    Triggers flashcard + quiz generation on demand, AFTER notes are ready.
    Generation runs as a background asyncio Task so the request returns
    immediately.  Poll GET /status or GET /generate/flashcards/{stem}/status
    to track progress.

  ③ New endpoint:  GET /generate/flashcards/{video_stem}/status
    Returns {"state": "pending|running|done|failed", "flashcard_count": N,
             "quiz_count": N, "error": null|str}

  ④ GET /results/flashcards/{stem} and GET /results/quiz/{stem} are
    unchanged — they still return the generated data once it exists.

  ⑤ prompt_cards_from_notes() added to prompts.py — combines study-notes
    Markdown + raw concepts + transcript into one focused call that fits
    comfortably within the 4 096-token Phi-3-mini context window.

Outputs per video
  ① JSON API         GET /results/video
  ② Markdown notes   GET /results/notes/{stem}
  ③ PDF report       GET /results/pdf/{stem}        ← file download
  ④ Q&A Flashcards   GET /results/flashcards/{stem} ← on-demand (POST /generate/flashcards/{stem})
  ⑤ MCQ Quiz         GET /results/quiz/{stem}       ← on-demand (same trigger)
  ⑥ Knowledge Graph  GET /results/graph/{stem}

Student progress (in-memory + JSON on disk, no database)
  PATCH  /flashcards/{video_stem}/{card_index}/review
  GET    /progress/{video_stem}
  GET    /progress/{video_stem}/due

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
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

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
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
    WHISPER_AVAILABLE,
    FASTER_WHISPER_AVAILABLE,   # v3.2.0: faster-whisper backend flag
    OPENAI_WHISPER_AVAILABLE,   # v3.2.0: openai-whisper fallback flag
)

# ── v3 additions ──────────────────────────────────────────────────────────────
from academic_system.slide_detector   import SlideChangeDetector
from academic_system.deduplicator     import SemanticDeduplicator
from academic_system.knowledge_graph  import KnowledgeGraphBuilder
from academic_system.language_support import LanguageDetector

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  v3.1.0: LONG-VIDEO CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# Maximum frames passed to LLM prompts in Phase 2 (prevents token overflow)
_PHASE2_MAX_FRAMES: int = int(os.environ.get("PHASE2_MAX_FRAMES", "40"))

# Adaptive FPS thresholds (seconds) — keeps frame count manageable
_FPS_LONG_THRESHOLD_1: int = int(os.environ.get("FPS_LONG_THRESHOLD_1", str(20 * 60)))  # 20 min
_FPS_LONG_THRESHOLD_2: int = int(os.environ.get("FPS_LONG_THRESHOLD_2", str(40 * 60)))  # 40 min
_FPS_FOR_LONG_1: float     = float(os.environ.get("FPS_FOR_LONG_1", "0.2"))   # 1 frame/5s
_FPS_FOR_LONG_2: float     = float(os.environ.get("FPS_FOR_LONG_2", "0.1"))   # 1 frame/10s

# Hard cap on total frames extracted (guards against OOM for very long videos)
_MAX_FRAMES_EXTRACT: int = int(os.environ.get("MAX_FRAMES_EXTRACT", "720"))

# Whisper transcription timeout (seconds) — 10 min default
_WHISPER_TIMEOUT_SEC: int = int(os.environ.get("WHISPER_TIMEOUT_SEC", "600"))

# v3.1.0: default Whisper to tiny for speed; override with WHISPER_MODEL_SIZE=base
_WHISPER_MODEL_SIZE: str = os.environ.get("WHISPER_MODEL_SIZE", "tiny")


# ──────────────────────────────────────────────────────────────────────────────
#  APP
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Modal Academic Intelligence System",
    description=(
        "Transforms lecture videos, slide images, and audio recordings into "
        "structured student learning materials.\n\n"
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
    version="3.2.0",
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
_progress: Dict[str, Dict[str, List[Dict]]] = {}

# ── v3.0.3: per-stem flashcard generation tasks ───────────────────────────────
_flashcard_tasks:  Dict[str, asyncio.Task] = {}
_flashcard_states: Dict[str, Dict[str, Any]] = {}
_shared_llm: Optional[LlamaReasoner] = None

# ── v3.1.0: shared OCR executor for async dispatch ───────────────────────────
# Declared here; populated in run_academic_pipeline to match worker count.
_ocr_executor: Optional[Any] = None


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


def save_frame(
    frame: np.ndarray,
    frame_id: int,
    timestamp: float,
    video_path: str,
) -> Tuple[str, str]:
    frames_dir = _video_frames_dir(video_path)
    filename   = f"frame_{frame_id:05d}_t{timestamp:.3f}s.jpg"
    path       = os.path.join(frames_dir, filename)
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])  # v3.1.0: 90→85 for speed
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


# ── v3.1.0: Video duration helper ─────────────────────────────────────────────

def _get_video_duration_sec(video_path: str) -> Optional[float]:
    """
    Return video duration in seconds using OpenCV.
    Returns None if the file cannot be opened or duration is unavailable.
    Fast — opens the file header only, does not decode frames.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps_cv  = cap.get(cv2.CAP_PROP_FPS)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps_cv > 0 and n_frames > 0:
            return n_frames / fps_cv
        return None
    except Exception:
        return None


def _adaptive_fps(duration_sec: Optional[float]) -> float:
    """
    Return the extraction FPS appropriate for a given video duration.

    Duration        FPS    Max frames (60min)
    ──────────────  ─────  ──────────────────
    ≤ 20 min        0.5    600
    20–40 min       0.2    240  (with slide dedup: ~80–120 unique)
    > 40 min        0.1    360  (with slide dedup: ~60–100 unique)

    Hard cap _MAX_FRAMES_EXTRACT (default 720) is enforced in the pipeline
    regardless of this value.
    """
    if duration_sec is None:
        return config.fps  # unknown — use configured default

    if duration_sec > _FPS_LONG_THRESHOLD_2:
        return _FPS_LONG_THRESHOLD_2 and _FPS_FOR_LONG_2  # > 40 min
    if duration_sec > _FPS_LONG_THRESHOLD_1:
        return _FPS_FOR_LONG_1  # 20–40 min
    return config.fps  # ≤ 20 min


# ── LLM helpers ───────────────────────────────────────────────────────────────

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
    """Extract the first complete, brace-balanced JSON object from raw text."""
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
    """Recover complete JSON objects from a truncated JSON array string."""
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


# ── v3.1.0: Phase 2 frame sampling ───────────────────────────────────────────

def _sample_frames_for_phase2(frames: List[Dict], max_n: int = _PHASE2_MAX_FRAMES) -> List[Dict]:
    """
    Evenly sample up to max_n frames from the full list for LLM prompt input.

    For a 1-hour lecture at fps=0.1 with slide dedup, we may collect 200–400
    accepted frames.  Feeding all of them into prompt_metadata() would produce
    a 6000+ token prompt that overflows Phi-3-mini's 4096-token context window.

    This function ensures Phase 2 always receives a bounded, representative
    sample regardless of lecture length.
    """
    if len(frames) <= max_n:
        return frames
    step = len(frames) / max_n
    return [frames[int(i * step)] for i in range(max_n)]


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
#  ON-DEMAND FLASHCARD GENERATION  (v3.0.3)
# ──────────────────────────────────────────────────────────────────────────────

async def _run_flashcard_generation_with_llm(video_stem: str) -> None:
    """Reuses the pipeline's already-loaded LLM if available."""
    global _shared_llm
    state = _flashcard_states[video_stem]

    if _shared_llm is not None:
        logger.info(f"[Flashcards/{video_stem}] Reusing pipeline LLM — no reload needed.")
        await _run_flashcard_generation(video_stem, _shared_llm)
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

    await _run_flashcard_generation(video_stem, llm)


async def _run_flashcard_generation(video_stem: str, llm: LlamaReasoner) -> None:
    """Background task that generates flashcards + quiz for a given video stem."""
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

        lecture_summary: Dict = vr.get("lecture_summary", {})
        deduped_concepts: List[str] = vr.get("deduped_concepts", [])
        deduped_formulas: List[str] = vr.get("deduped_formulas", [])

        transcript: str = ""
        audio_analysis = vr.get("audio_analysis") or {}
        if isinstance(audio_analysis, dict):
            transcript = audio_analysis.get("text", "")

        prompt = prompt_cards_from_notes(
            notes_md         = notes_md,
            lecture_title    = lecture_summary.get("lecture_title", video_stem),
            subject_area     = lecture_summary.get("subject_area", "General"),
            key_concepts     = deduped_concepts,
            formulas         = deduped_formulas,
            transcript       = transcript,
            topics           = lecture_summary.get("main_topics", []),
            learning_outcomes= lecture_summary.get("learning_outcomes", []),
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

    except Exception as exc:
        logger.error(
            f"[Flashcards/{video_stem}] Generation FAILED: {exc}",
            exc_info=True,
        )
        state["state"] = "failed"
        state["error"] = str(exc)


# ──────────────────────────────────────────────────────────────────────────────
#  CORE ASYNC PIPELINE  (v3.1.0 — 1-hour video support)
# ──────────────────────────────────────────────────────────────────────────────

async def run_academic_pipeline(
    video_paths: List[str],
    video_fps_overrides: Optional[Dict[str, float]] = None,
) -> None:
    """
    Full v3.1.0 pipeline.

    Key changes over v3.0.3
    -----------------------
    • Adaptive FPS — long videos use a lower extraction rate (0.2 or 0.1 fps)
      to keep frame count under _MAX_FRAMES_EXTRACT.
    • Async OCR — _flush_ocr_batch runs in run_in_executor so the event loop
      is never blocked by EasyOCR's synchronous GPU calls.
    • Phase 2 frame cap — at most _PHASE2_MAX_FRAMES frames are fed to LLM
      prompts, preventing token overflow on 1-hour lectures.
    • Tiny Whisper — _WHISPER_MODEL_SIZE defaults to "tiny" for 4x faster
      transcription with acceptable accuracy on lecture audio.
    • Whisper timeout — transcription is bounded by _WHISPER_TIMEOUT_SEC.
    """
    global pipeline_running, academic_results, stream_frame_counters, _shared_llm

    pipeline_running      = True
    stream_frame_counters = {}

    device  = setup_device()

    # ── v3.1.0: apply per-video FPS overrides to StreamManager config ─────────
    # StreamManager reads config.fps at construction time, so we patch it
    # temporarily if we need a different rate for this batch.
    effective_fps = config.fps
    if video_fps_overrides:
        # Use the lowest FPS required across all videos in this batch
        lowest_fps = min(video_fps_overrides.values())
        if lowest_fps < config.fps:
            effective_fps = lowest_fps
            logger.info(
                f"[Pipeline v3.1.0] Adaptive FPS: using {effective_fps:.2f} fps "
                f"(config default is {config.fps} fps)."
            )

    # Temporarily override config.fps for StreamManager construction
    _original_fps = config.fps
    config.fps    = effective_fps

    sources = {f"stream_{i}": p for i, p in enumerate(video_paths)}

    import concurrent.futures
    import threading

    stream_manager = StreamManager(sources, target_fps=effective_fps)

    # Restore config.fps immediately after StreamManager is constructed
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

    # ── v3.1.0: dedicated ThreadPoolExecutor for async OCR dispatch ───────────
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
                raise RuntimeError(
                    "OCRExtractor returned None — check SSL/certificate errors above."
                )
            if not callable(getattr(ocr_extractor, "extract", None)):
                raise RuntimeError(
                    "OCRExtractor does not expose .extract() — initialisation failed."
                )

            logger.info("Models ready ✓ (Phi-3 + EasyOCR)")

            global _shared_llm
            _shared_llm = llm

        except Exception as exc:
            _model_error[0] = exc
            logger.error(
                f"Model loading FAILED: {exc}\n"
                "If this is an SSL error run:  pip install --upgrade certifi\n"
                "Then restart the server.",
                exc_info=True,
            )
        finally:
            _models_ready.set()

    _model_thread = threading.Thread(
        target=_load_models, daemon=True, name="model_loader"
    )
    _model_thread.start()

    _save_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="jpeg_save"
    )

    _executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=len(video_paths),
        thread_name_prefix="whisper",
    )

    def _run_whisper(vp: str) -> Dict:
        """
        v3.1.0: uses _WHISPER_MODEL_SIZE (default "tiny") and enforces
        _WHISPER_TIMEOUT_SEC timeout per video.
        """
        try:
            audio_path = academic_results[vp].get("_audio_path")
        except KeyError:
            logger.error(
                f"[Whisper/bg] academic_results missing for {vp} — "
                "no transcript will be available."
            )
            return {}
        if not audio_path or not os.path.isfile(audio_path):
            return {}
        try:
            result = transcribe(
                audio_path,
                language   = None,
                model_size = _WHISPER_MODEL_SIZE,   # v3.1.0: "tiny" default
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
    # v3.1.0: larger default batch size for GPU efficiency
    _OCR_BATCH = int(os.environ.get("OCR_BATCH_SIZE", "8"))
    _pending_ocr:      List[tuple] = []
    _pre_model_buffer: List[tuple] = []

    # ── v3.1.0: async OCR flush — runs EasyOCR in thread pool ────────────────

    def _flush_ocr_batch_sync(batch: List[tuple]) -> None:
        """
        Synchronous OCR + LLM processing for one batch.
        Runs inside _ocr_thread_pool via asyncio.get_event_loop().run_in_executor.
        """
        if not batch:
            return

        if _model_error[0] is not None:
            raise RuntimeError(
                f"Cannot run OCR — model loading failed earlier: {_model_error[0]}\n"
                "Fix the error above, then restart the server and re-upload."
            )

        if ocr_extractor is None:
            raise RuntimeError(
                "ocr_extractor is None — EasyOCR did not initialise correctly."
            )

        raw_frames = [item[0] for item in batch]
        try:
            all_ocr = ocr_extractor.batch_extract(
                raw_frames, config.ocr_confidence_threshold
            )
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

            # Skip LLM entirely — just tag importance by OCR length
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
        """
        v3.1.0: Async wrapper — dispatches synchronous OCR to thread pool
        so the asyncio event loop is never blocked by EasyOCR GPU calls.

        Pre-model buffer path: if models are not ready yet, we buffer frames
        (same as before) and drain them once models are loaded.
        """
        if not batch:
            return

        if not _models_ready.is_set():
            logger.debug(
                f"[OCR] Models not ready — buffering {len(batch)} frame(s) "
                "for re-processing once models load."
            )
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
        logger.info(
            f"[OCR] Draining {len(_pre_model_buffer)} pre-model-load frame(s) "
            "now that models are ready."
        )
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

            # v3.1.0: hard cap on total extracted frames per stream
            for i, stream_id in enumerate(stream_ids):
                frame      = frames[i]
                timestamp  = float(timestamps[i])
                video_path = stream_id_to_path[stream_id]
                stats      = slide_stats[stream_id]

                # Hard cap: stop accepting new frames once limit is reached
                current_count = stream_frame_counters.get(stream_id, 0)
                if current_count >= _MAX_FRAMES_EXTRACT:
                    if current_count == _MAX_FRAMES_EXTRACT:
                        logger.warning(
                            f"[{stream_id}] Frame cap reached: {_MAX_FRAMES_EXTRACT} frames. "
                            f"Remaining video will not be extracted (timestamp={timestamp:.1f}s). "
                            f"Lower PIPELINE_FPS or raise MAX_FRAMES_EXTRACT to change this."
                        )
                    stats["frames_skipped"] += 1
                    continue

                stats["frames_seen"] += 1

                if not slide_detectors[stream_id].is_new_slide(frame, timestamp):
                    stats["frames_skipped"] += 1
                    logger.debug(
                        f"[{stream_id}] @{timestamp:.1f}s — duplicate, skipped."
                    )
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
            logger.info(
                "Pipeline stopped early — generating outputs from collected frames."
            )

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

        # ── Collect Whisper results with timeout ──────────────────────────────
        # v3.2.0 note: faster-whisper enforces its OWN internal timeouts
        # (_TOTAL_TIMEOUT_SEC and per-chunk _CHUNK_TIMEOUT_SEC) inside transcribe().
        # The outer future.result() timeout here is a last-resort safety net ONLY —
        # it should be larger than the internal timeout so it never fires during
        # normal faster-whisper partial-result collection.
        # Using _WHISPER_TIMEOUT_SEC + 120 (2 min grace period) as the outer limit.
        _outer_timeout = _WHISPER_TIMEOUT_SEC + 120
        whisper_results: Dict[str, Dict] = {}
        for vp, future in whisper_futures.items():
            try:
                whisper_results[vp] = future.result(timeout=_outer_timeout)
            except concurrent.futures.TimeoutError:
                logger.error(
                    f"[Whisper] Outer timeout ({_outer_timeout}s) for {vp} — "
                    "transcriber thread appears completely stuck. "
                    "Pipeline will continue without transcript."
                )
                whisper_results[vp] = {"error": "timeout", "text": "", "segments": []}
            except Exception as exc:
                logger.error(f"Whisper future failed for {vp}: {exc}")
                whisper_results[vp] = {}
        _executor.shutdown(wait=False)
        _save_executor.shutdown(wait=True)
        
        if not _models_ready.is_set():
            logger.info(
                "Waiting for model loading to complete before Phase 2…"
            )
            _models_ready.wait()

        if _model_error[0] is None:
            await _drain_pre_model_buffer()
        _ocr_thread_pool.shutdown(wait=False)  # v3.1.0: shutdown async OCR pool

        

        if _model_error[0] is not None:
            logger.error(
                f"[Phase 2 ABORTED] Model load failed: {_model_error[0]}\n"
                "All video results will be empty.\n"
                "Fix the error above, restart the server, and re-upload the video."
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
                    f"[{stem}] ALL {len(frames_list)} frames have empty OCR / "
                    f"importance=low. Phase 2 will rely on transcript only.\n"
                    "If this is unexpected, check that EasyOCR loaded correctly."
                )
                frames_for_phase2 = frames_list
            else:
                logger.info(
                    f"[{stem}] {len(meaningful_frames)}/{len(frames_list)} frames "
                    f"have meaningful OCR content."
                )
                frames_for_phase2 = frames_list

            # ── v3.1.0: Phase 2 frame cap ─────────────────────────────────────
            # Sample at most _PHASE2_MAX_FRAMES frames for LLM prompts.
            # This is the critical guard against token overflow on 1-hour videos.
            frames_for_llm = _sample_frames_for_phase2(frames_for_phase2, _PHASE2_MAX_FRAMES)
            if len(frames_for_llm) < len(frames_for_phase2):
                logger.info(
                    f"[{stem}] Phase 2 frame sampling: {len(frames_for_phase2)} → "
                    f"{len(frames_for_llm)} frames for LLM prompts "
                    f"(cap={_PHASE2_MAX_FRAMES})."
                )

            # ── Whisper result ────────────────────────────────────────────────
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
                logger.info(
                    f"Language: {lang_info['name']} (code={lang_info['code']})"
                )
                ocr_extractor.set_languages(lang_info["ocr_langs"])
            else:
                vr["audio_analysis"] = transcription or None

            def _patch(prompt: str) -> str:
                return _lang_detector.patch_prompt(prompt, lang_info)

            # ── TWO-CALL PHASE 2 ──────────────────────────────────────────────
            # Call 1: metadata  (~300 tok output)  — uses sampled frames
            # Call 2: study notes Markdown  (~700 tok output)
            # ─────────────────────────────────────────────────────────────────

            logger.info("Phase 2: 2-call split — Call 1 (metadata)…")

            audio_topics:    Dict[str, Any] = {}
            lecture_summary: Dict[str, Any] = {}
            notes_md   = ""
            meta:       Dict[str, Any] = {}

            # ── Call 1: metadata (uses sampled frames) ────────────────────────
            try:
                call1_prompt = _patch(
                    prompt_metadata(
                        video_path,
                        frames_for_llm,   # v3.1.0: sampled, not full list
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
                    f"Call 1 done: title='{meta.get('lecture_title','?')[:50]}' "
                    f"concepts={meta.get('key_concepts',[])} "
                    f"topics={meta.get('topics',[])}"
                )
            else:
                logger.warning(
                    "Call 1 returned empty metadata — Call 2 will use defaults."
                )

            # ── Call 2: study notes ───────────────────────────────────────────
            logger.info("Phase 2: 2-call split — Call 2 (study notes)…")
            raw_formulas: List[str] = list({
                f
                for fr in frames_for_llm   # v3.1.0: sampled frames
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

            # Fallback minimal notes if generation failed/truncated
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

            # ── Deduplication (uses full frame list for accuracy) ─────────────
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
                f"{len(deduped_concepts)} concepts, "
                f"{len(deduped_formulas)} formulas."
            )

            # ── Persist notes ─────────────────────────────────────────────────
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

            # ── PDF + knowledge graph concurrently ────────────────────────────
            def _build_graph():
                try:
                    graph    = _graph_builder.build(
                        frames_list, audio_topics, lecture_summary
                    )
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

            logger.info(
                f"[{stem}] Pipeline complete. "
                f"Flashcards not yet generated — call "
                f"POST /generate/flashcards/{stem} when ready."
            )

        pipeline_running = False
        logger.info(
            "Academic pipeline v3.1.0 complete. "
            "Notes and PDF are ready. "
            "Use POST /generate/flashcards/{stem} to generate flashcards and quiz."
        )


# ──────────────────────────────────────────────────────────────────────────────
#  PYDANTIC MODELS
# ──────────────────────────────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    confidence: int           = Field(..., ge=1, le=5, description="1=very hard → 5=very easy")
    correct:    bool          = Field(False, description="Did the student answer correctly?")
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

    for key in [
        k for k, v in academic_results.items() if v.get("input_type") == "video"
    ]:
        del academic_results[key]

    video_paths: List[str] = []
    video_fps_overrides:  Dict[str, float] = {}  # v3.1.0

    for upload in uploaded:
        dest = os.path.join(UPLOAD_DIR, upload.filename)
        with open(dest, "wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        logger.info(f"Video saved → {dest}")

        meta       = extract_video_metadata(dest)
        audio_path = extract_audio(dest, AUDIO_DIR)
        frames_dir = _video_frames_dir(dest)

        # ── v3.1.0: detect duration and set adaptive FPS ──────────────────────
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
                    f"[{Path(dest).stem}] Video is longer than 60 minutes "
                    f"({duration_min:.1f} min). Processing will be capped at "
                    f"{_MAX_FRAMES_EXTRACT} frames. Consider trimming to ≤60 min "
                    "for best results."
                )
        else:
            logger.info(
                f"[{Path(dest).stem}] Duration unknown — using default FPS {adaptive_fps:.2f}"
            )

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
            # v3.1.0 additions
            "duration_sec":          duration_sec,
            "adaptive_fps":          adaptive_fps,
            "whisper_model":         _WHISPER_MODEL_SIZE,
        }
        video_paths.append(dest)

    pipeline_task = asyncio.create_task(
        run_academic_pipeline(video_paths, video_fps_overrides=video_fps_overrides)
    )

    # Build duration info for response
    duration_info = {}
    for vp in video_paths:
        vr = academic_results[vp]
        dur = vr.get("duration_sec")
        duration_info[Path(vp).name] = {
            "duration_min": round(dur / 60, 1) if dur else "unknown",
            "adaptive_fps": vr.get("adaptive_fps", config.fps),
            "max_frames":   _MAX_FRAMES_EXTRACT,
            "whisper_model": _WHISPER_MODEL_SIZE,
        }

    return JSONResponse({
        "message": (
            f"Academic pipeline v3.1.0 started for {len(video_paths)} video(s). "
            "Notes and PDF will be generated automatically. "
            "Flashcards and quiz require a separate call to "
            "POST /generate/flashcards/{stem} after notes are ready."
        ),
        "videos":       video_paths,
        "video_config": duration_info,
        "poll":         "GET /status",
        "outputs": {
            "json":            "GET /results/video",
            "study_notes":     "GET /results/notes/{stem}",
            "pdf":             "GET /results/pdf/{stem}",
            "flashcards":      "POST /generate/flashcards/{stem}  ← trigger first",
            "flashcards_get":  "GET /results/flashcards/{stem}    ← retrieve after",
            "quiz_get":        "GET /results/quiz/{stem}",
            "knowledge_graph": "GET /results/graph/{stem}",
            "frames":          "GET /results/frames/{stem}",
        },
        "v310_changes": {
            "adaptive_fps":    "Long videos auto-reduce FPS to stay under frame cap",
            "tiny_whisper":    f"Whisper model: {_WHISPER_MODEL_SIZE} (4x faster than base)",
            "async_ocr":       "OCR runs in thread pool — event loop never blocked",
            "phase2_cap":      f"Max {_PHASE2_MAX_FRAMES} frames fed to LLM in Phase 2",
            "frame_cap":       f"Hard cap: {_MAX_FRAMES_EXTRACT} frames extracted per video",
        },
    })


@app.post(
    "/upload/image",
    summary="Upload 1–3 slide or diagram images",
    tags=["Upload"],
)
async def upload_image(
    file1:    UploadFile = File(None),
    file2:    UploadFile = File(None),
    file3:    UploadFile = File(None),
    language: str        = Query(
        "en",
        description=(
            "ISO 639-1 language code for OCR (e.g. hi, zh, ar). "
            "GET /languages for full list."
        ),
    ),
) -> JSONResponse:
    uploaded = [f for f in (file1, file2, file3) if f is not None]
    if not uploaded:
        raise HTTPException(400, "At least one image file is required.")
    for f in uploaded:
        _assert_ext(f.filename, IMAGE_EXTENSIONS, "image")

    lang_info = _lang_detector.from_code(language)
    device    = setup_device()
    ocr       = OCRExtractor(
        use_gpu   = (device == "cuda"),
        languages = lang_info["ocr_langs"],
    )
    llm = LlamaReasoner(
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


@app.post(
    "/upload/audio",
    summary="Upload 1–3 lecture audio files",
    tags=["Upload"],
)
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

        ext             = Path(dest).suffix.lower()
        processing_path = convert_to_wav(dest, AUDIO_DIR) if ext != ".wav" else dest

        try:
            transcription = transcribe(
                processing_path,
                language   = None,
                model_size = _WHISPER_MODEL_SIZE,   # v3.1.0: uses env-var default
            )
            lang_info = _lang_detector.from_whisper(transcription)
            entry = {
                "input_type":        "audio",
                "audio_path":        dest,
                "detected_language": {
                    "code": lang_info["code"],
                    "name": lang_info["name"],
                },
                "transcription":  transcription,
                "whisper_model":  _WHISPER_MODEL_SIZE,
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
        "whisper_model":     _WHISPER_MODEL_SIZE,
        "note":              "Language is auto-detected by Whisper.",
    })


# ──────────────────────────────────────────────────────────────────────────────
#  ON-DEMAND FLASHCARD GENERATION  (v3.0.3, unchanged)
# ──────────────────────────────────────────────────────────────────────────────

@app.post(
    "/generate/flashcards/{video_stem}",
    summary="Trigger on-demand flashcard + quiz generation from saved notes",
    tags=["Flashcards & Quiz"],
)
async def generate_flashcards(video_stem: str) -> JSONResponse:
    vr = _find_video(video_stem)
    if vr is None:
        notes_path = os.path.join(NOTES_DIR, f"{video_stem}_study_notes.md")
        if not os.path.isfile(notes_path):
            raise HTTPException(
                404,
                f"No pipeline result found for '{video_stem}'. "
                "Upload the video first and wait for the pipeline to complete.",
            )
    else:
        if not vr.get("study_notes") and not os.path.isfile(
            os.path.join(NOTES_DIR, f"{video_stem}_study_notes.md")
        ):
            raise HTTPException(
                503,
                "Study notes are not yet ready. "
                "Wait for study_notes_ready=true in GET /status, then retry.",
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

    task = asyncio.create_task(_run_flashcard_generation_with_llm(video_stem))
    _flashcard_tasks[video_stem] = task

    return JSONResponse(
        status_code=202,
        content={
            "message": (
                f"Flashcard generation started for '{video_stem}'. "
                "Poll the status endpoint to track progress."
            ),
            "state": "pending",
            "poll":  f"GET /generate/flashcards/{video_stem}/status",
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
async def flashcard_generation_status(video_stem: str) -> JSONResponse:
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
#  CONTROL
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/stop", summary="Stop the running pipeline", tags=["Control"])
async def stop_pipeline() -> JSONResponse:
    global pipeline_running
    pipeline_running = False
    return JSONResponse({
        "message": "Stop signal sent. Outputs generated from collected frames."
    })


@app.delete("/results", summary="Clear all in-memory results", tags=["Control"])
async def clear_results() -> JSONResponse:
    academic_results.clear()
    _flashcard_states.clear()
    _flashcard_tasks.clear()
    return JSONResponse({"message": "All in-memory results cleared."})


# ──────────────────────────────────────────────────────────────────────────────
#  STATUS
# ──────────────────────────────────────────────────────────────────────────────

@app.get(
    "/status",
    summary="Pipeline progress and per-video readiness",
    tags=["Status"],
)
async def status() -> JSONResponse:
    videos = {
        p: v for p, v in academic_results.items()
        if v.get("input_type") == "video"
    }
    images = {
        p: v for p, v in academic_results.items()
        if v.get("input_type") == "image"
    }
    audios = {
        p: v for p, v in academic_results.items()
        if v.get("input_type") == "audio"
    }

    video_states = {}
    for p, v in videos.items():
        ss     = v.get("slide_change_stats", {})
        lang   = v.get("detected_language") or {}
        stem   = Path(p).stem
        fc_st  = _flashcard_states.get(stem, {})
        dur    = v.get("duration_sec")
        video_states[os.path.basename(p)] = {
            "duration_min":             round(dur / 60, 1) if dur else None,
            "adaptive_fps":             v.get("adaptive_fps"),
            "whisper_model":            v.get("whisper_model", _WHISPER_MODEL_SIZE),
            "frames_seen":              ss.get("frames_seen", 0),
            "unique_slides_accepted":   ss.get(
                "slides_accepted", len(v["per_frame_details"])
            ),
            "duplicate_frames_skipped": ss.get("frames_skipped", 0),
            "frames_collected":         len(v["per_frame_details"]),
            "frame_cap":                _MAX_FRAMES_EXTRACT,
            "phase2_frame_cap":         _PHASE2_MAX_FRAMES,
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
            "flashcard_generate_url":   f"POST /generate/flashcards/{stem}",
        }

    return JSONResponse({
        "pipeline_running":       pipeline_running,
        "task_done":              pipeline_task.done() if pipeline_task else True,
        "total_frames_collected": sum(
            len(v["per_frame_details"]) for v in videos.values()
        ),
        "videos_in_pipeline":   len(videos),
        "images_analysed":      len(images),
        "audio_files_analysed": len(audios),
        "videos":               video_states,
    })


# ──────────────────────────────────────────────────────────────────────────────
#  RESULTS  (unchanged from v3.0.3)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/results/video",    summary="Full JSON results for all videos",        tags=["Results"])
async def results_video() -> List[Dict[str, Any]]:
    return [v for v in academic_results.values() if v.get("input_type") == "video"]


@app.get("/results/image",    summary="Analysis results for all images",         tags=["Results"])
async def results_image() -> List[Dict[str, Any]]:
    return [v for v in academic_results.values() if v.get("input_type") == "image"]


@app.get("/results/audio",    summary="Transcription results for all audio",     tags=["Results"])
async def results_audio() -> List[Dict[str, Any]]:
    return [v for v in academic_results.values() if v.get("input_type") == "audio"]


@app.get("/results/notes/{video_stem}", summary="Markdown study notes",          tags=["Results"])
async def results_notes(video_stem: str) -> PlainTextResponse:
    v = _find_video(video_stem)
    if v:
        notes = v.get("study_notes")
        if notes:
            return PlainTextResponse(notes, media_type="text/markdown")
        err = v.get("error")
        if err:
            raise HTTPException(500, f"Pipeline error: {err}")
        raise HTTPException(503, "Study notes not yet generated — check /status.")
    p = os.path.join(NOTES_DIR, f"{video_stem}_study_notes.md")
    if os.path.isfile(p):
        return PlainTextResponse(open(p, encoding="utf-8").read(), media_type="text/markdown")
    raise HTTPException(404, f"No study notes found for '{video_stem}'.")


@app.get("/results/pdf/{video_stem}", summary="Download PDF academic report",    tags=["Results"])
async def results_pdf(video_stem: str) -> FileResponse:
    v = _find_video(video_stem)
    if v:
        pdf_path = v.get("pdf_report_path")
        if pdf_path and os.path.isfile(pdf_path):
            return FileResponse(pdf_path, media_type="application/pdf",
                                filename=os.path.basename(pdf_path))
        err = v.get("error")
        if err:
            raise HTTPException(500, f"Pipeline error: {err}")
        raise HTTPException(503, "PDF not yet generated — check /status.")
    p = os.path.join(PDF_DIR, f"{video_stem}_academic_report.pdf")
    if os.path.isfile(p):
        return FileResponse(p, media_type="application/pdf", filename=os.path.basename(p))
    raise HTTPException(404, f"No PDF found for '{video_stem}'.")


@app.get(
    "/results/flashcards/{video_stem}",
    summary="Q&A flashcards (requires POST /generate/flashcards/{stem} first)",
    tags=["Flashcards & Quiz"],
)
async def results_flashcards(video_stem: str) -> List[Dict]:
    v = _find_video(video_stem)
    if v:
        cards = v.get("flashcards")
        if cards:
            return cards
        fc_state = _flashcard_states.get(video_stem, {}).get("state", "idle")
        if fc_state == "running":
            raise HTTPException(503, f"Flashcards are still being generated. "
                                     f"Poll GET /generate/flashcards/{video_stem}/status.")
        if fc_state == "idle":
            raise HTTPException(404, f"Flashcards not yet generated for '{video_stem}'. "
                                     f"Call POST /generate/flashcards/{video_stem} first.")
        err = _flashcard_states.get(video_stem, {}).get("error")
        if err:
            raise HTTPException(500, f"Flashcard generation error: {err}")

    p = os.path.join(FLASHCARD_DIR, f"{video_stem}_flashcards.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            cards = json.load(f)
        if cards:
            return cards

    raise HTTPException(404, f"No flashcards found for '{video_stem}'. "
                             f"Call POST /generate/flashcards/{video_stem} first.")


@app.get(
    "/results/quiz/{video_stem}",
    summary="MCQ quiz (requires POST /generate/flashcards/{stem} first)",
    tags=["Flashcards & Quiz"],
)
async def results_quiz(video_stem: str) -> List[Dict]:
    v = _find_video(video_stem)
    if v:
        quiz = v.get("quiz")
        if quiz:
            return quiz
        fc_state = _flashcard_states.get(video_stem, {}).get("state", "idle")
        if fc_state == "running":
            raise HTTPException(503, f"Quiz is still being generated. "
                                     f"Poll GET /generate/flashcards/{video_stem}/status.")
        if fc_state == "idle":
            raise HTTPException(404, f"Quiz not yet generated for '{video_stem}'. "
                                     f"Call POST /generate/flashcards/{video_stem} first.")
        err = _flashcard_states.get(video_stem, {}).get("error")
        if err:
            raise HTTPException(500, f"Quiz generation error: {err}")

    p = os.path.join(QUIZ_DIR, f"{video_stem}_quiz.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            quiz = json.load(f)
        if quiz:
            return quiz

    raise HTTPException(404, f"No quiz found for '{video_stem}'. "
                             f"Call POST /generate/flashcards/{video_stem} first.")


@app.get("/results/graph/{video_stem}", summary="Knowledge graph in D3.js format", tags=["Results"])
async def results_graph(video_stem: str) -> Dict:
    v = _find_video(video_stem)
    if v:
        kg = v.get("knowledge_graph")
        if kg is not None:
            return kg
        err = v.get("error")
        if err:
            raise HTTPException(500, f"Pipeline error: {err}")
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
#  STUDENT PROGRESS API  (unchanged from v3.0.3)
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
    global _progress

    v = _find_video(video_stem)
    if v:
        flashcards = v.get("flashcards", [])
        if card_index < 0 or card_index >= len(flashcards):
            raise HTTPException(
                404,
                f"Card index {card_index} out of range "
                f"(0–{len(flashcards) - 1} for this video).",
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
    session_id: Optional[str] = Query(None),
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
#  LANGUAGE INFO
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/languages", summary="Supported OCR + transcription languages", tags=["Info"])
async def list_languages() -> Dict:
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
        "system":    "Multi-Modal Academic Intelligence System v3.2.0",
        "reasoning": config.reasoning_model_id,
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
            "env_overrides": {
                "WHISPER_MODEL_SIZE":     "tiny|base|small|medium|large",
                "WHISPER_TIMEOUT_SEC":    "seconds (default 600)",
                "MAX_FRAMES_EXTRACT":     "hard cap per video (default 720)",
                "PHASE2_MAX_FRAMES":      "frames fed to LLM (default 40)",
                "FPS_LONG_THRESHOLD_1":   "seconds for long-1 FPS switch (default 1200)",
                "FPS_LONG_THRESHOLD_2":   "seconds for long-2 FPS switch (default 2400)",
                "FPS_FOR_LONG_1":         "FPS for 20-40min videos (default 0.2)",
                "FPS_FOR_LONG_2":         "FPS for >40min videos (default 0.1)",
                "OCR_BATCH_SIZE":         "frames per EasyOCR call (default 8)",
                "OCR_WORKERS":            "async OCR thread workers (default 2)",
            },
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
            "easyocr":                    EASYOCR_AVAILABLE,
            "tesseract":                  TESSERACT_AVAILABLE,
            "whisper_any":                WHISPER_AVAILABLE,
            "whisper_faster_whisper":     FASTER_WHISPER_AVAILABLE,   # v3.2.0
            "whisper_openai_fallback":    OPENAI_WHISPER_AVAILABLE,   # v3.2.0
            "whisper_active_backend":     (                            # v3.2.0
                "faster-whisper" if FASTER_WHISPER_AVAILABLE
                else "openai-whisper" if OPENAI_WHISPER_AVAILABLE
                else "none"
            ),
            "whisper_model":              _WHISPER_MODEL_SIZE,
            "skimage_ssim":               SKIMAGE_AVAILABLE,
            "sentence_transformers":      ST_AVAILABLE,
            "sklearn_tfidf":              SKLEARN_AVAILABLE,
            "networkx":                   NX_AVAILABLE,
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
            "⑤ MCQ Quiz     GET /results/quiz/{stem}          ← retrieve (same trigger)",
            "⑥ Graph (D3)   GET /results/graph/{stem}",
        ],
        "all_endpoints": {
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
            "languages":            "GET    /languages",
            "stop":                 "POST   /stop",
            "clear":                "DELETE /results",
            "docs":                 "GET    /docs",
            "diagnostics":          "GET    /diagnostics",
        },
    })