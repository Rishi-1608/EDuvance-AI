"""
Live Lecture Module  —  v1.0.0
==============================
Real-time audio transcription via WebSocket + on-session-end pipeline trigger.

New endpoints
─────────────
  POST  /live/start                    → create session, return session_id
  WS    /live/lecture/{session_id}     → stream audio in, receive segments back
  POST  /live/end/{session_id}         → stop, trigger Phase-2 academic pipeline
  GET   /live/status/{session_id}      → session state + pipeline progress
  GET   /live/transcript/{session_id}  → full accumulated transcript
  GET   /live/sessions                 → list user's sessions

How it plugs into the existing pipeline (zero duplication)
──────────────────────────────────────────────────────────
  When POST /live/end/{id} is called (or WS disconnects):
    1. Final Whisper flush on all buffered audio chunks.
    2. academic_results[pipeline_stem] is seeded so all GET /results/* work.
    3. _run_phase2_for_live() calls the same Phi-3 prompts as the video pipeline:
         Call 1 → metadata (prompt_metadata)
         Call 2 → study notes (prompt_study_notes_text)
         Phase 3 → knowledge graph + PDF (parallel)
    4. POST /generate/flashcards/{stem} works unchanged on-demand.
    5. DB persistence via save_pipeline_result_to_db (same as video pipeline).

Mount in main_v7-9.py
──────────────────────
  from live_lecture import router as live_router
  app.include_router(live_router)

Audio format
────────────
  Browser sends MediaRecorder chunks (WebM/Opus).
  ffmpeg merges all chunks into 16 kHz mono WAV for faster-whisper.
  Transcription runs every LIVE_CHUNK_INTERVAL_SEC seconds (default 4 s).

Authentication
──────────────
  REST endpoints: Bearer token in Authorization header (same as all other endpoints).
  WebSocket: pass JWT as query param: ws://.../live/lecture/{id}?token=<JWT>
  (browsers cannot set Authorization headers on WebSocket connections)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import auth
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

log = logging.getLogger(__name__)

router = APIRouter(prefix="/live", tags=["Live Lecture"])

# ── Tunables (override via environment variables) ─────────────────────────────
LIVE_CHUNK_INTERVAL_SEC: float = float(os.environ.get("LIVE_CHUNK_INTERVAL_SEC", "15"))
LIVE_WHISPER_MODEL:      str   = os.environ.get("LIVE_WHISPER_MODEL", "tiny")
LIVE_AUDIO_DIR:          str   = os.environ.get("LIVE_AUDIO_DIR",     "live_audio")
LIVE_MIN_CHUNK_BYTES:    int   = int(os.environ.get("LIVE_MIN_CHUNK_BYTES", "3200"))

os.makedirs(LIVE_AUDIO_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Session state machine
#  States: created → recording → processing → done | error
# ─────────────────────────────────────────────────────────────────────────────

class LiveSession:
    def __init__(self, session_id: str, user_id: int, title: str) -> None:
        self.session_id:        str   = session_id
        self.user_id:           int   = user_id
        self.title:             str   = title
        self.state:             str   = "created"
        self.started_at:        float = time.time()
        self.ended_at:          Optional[float] = None

        # Audio accumulation
        self.audio_dir:   str       = os.path.join(LIVE_AUDIO_DIR, session_id)
        self.chunk_index: int       = 0
        self.chunk_paths: List[str] = []
        self.merged_path: Optional[str] = None

        # Single growing WebM stream file (MediaRecorder fragments share headers)
        self._stream_webm_path: str = os.path.join(self.audio_dir, "stream.webm")
        self._prev_wav_duration: float = 0.0  # duration (sec) already extracted

        # Transcript
        self.segments:  List[Dict[str, Any]] = []
        self.full_text: str = ""
        self.last_transcribed_idx: int = -1  # Index of the last chunk transcribed

        # Pipeline bridge
        # stem format: "{user_id}_{session_id}" — _find_any_lecture() resolves by stem
        self.pipeline_stem:    str  = f"{user_id}_{session_id}"
        self.pipeline_started: bool = False
        self.pipeline_error:   Optional[str] = None

        self._lock = threading.Lock()
        os.makedirs(self.audio_dir, exist_ok=True)

    def add_segment(self, seg: Dict[str, Any]) -> bool:
        """Add a segment if not already present. Returns True if new."""
        with self._lock:
            existing = {s["start"] for s in self.segments}
            if seg.get("start") in existing:
                return False
            self.segments.append(seg)
            self.segments.sort(key=lambda s: s["start"])
            self.full_text = " ".join(s["text"].strip() for s in self.segments)
            return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id":       self.session_id,
            "user_id":          self.user_id,
            "title":            self.title,
            "state":            self.state,
            "started_at":       self.started_at,
            "ended_at":         self.ended_at,
            "duration_sec":     round((self.ended_at or time.time()) - self.started_at, 1),
            "segment_count":    len(self.segments),
            "word_count":       len(self.full_text.split()),
            "pipeline_stem":    self.pipeline_stem,
            "pipeline_started": self.pipeline_started,
            "pipeline_error":   self.pipeline_error,
        }


# Module-level session registry (mirrors academic_results pattern in main)
_live_sessions: Dict[str, LiveSession] = {}


# ─────────────────────────────────────────────────────────────────────────────
#  Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_wav_duration(wav_path: str) -> float:
    """Get duration of a WAV file in seconds via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _save_chunk(session: LiveSession, data: bytes) -> Optional[str]:
    """
    Append raw audio bytes to the single growing stream.webm file,
    then extract only the NEW portion as a standalone WAV chunk.

    MediaRecorder sends a continuous WebM stream — only the first
    ondataavailable blob carries the WebM header (EBML + Segment +
    Tracks).  Subsequent blobs are Cluster continuations that cannot
    be decoded in isolation.  By appending to one file we keep the
    container valid for ffmpeg at all times.
    """
    try:
        # 1. Append bytes to the single growing WebM stream
        with open(session._stream_webm_path, "ab") as fh:
            fh.write(data)

        wav_path = os.path.join(session.audio_dir, f"chunk_{session.chunk_index:05d}.wav")

        # 2. Convert the ENTIRE stream to a temp WAV to find its total duration
        full_wav = os.path.join(session.audio_dir, "_full_tmp.wav")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", session._stream_webm_path,
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", full_wav],
            capture_output=True, timeout=60,
        )
        if result.returncode != 0:
            stderr_msg = result.stderr.decode(errors='replace')[:300]
            log.error(f"[Live/{session.session_id}] stream→WAV failed: {stderr_msg}")
            return None

        total_dur = _get_wav_duration(full_wav)
        prev_dur  = session._prev_wav_duration
        new_dur   = total_dur - prev_dur

        if new_dur < 0.3:
            # Almost no new audio — skip this chunk
            os.remove(full_wav)
            return None

        # 3. Extract only the NEW audio slice
        result = subprocess.run(
            ["ffmpeg", "-y",
             "-ss", f"{prev_dur:.3f}",
             "-i", full_wav,
             "-c:a", "pcm_s16le",
             wav_path],
            capture_output=True, timeout=30,
        )
        os.remove(full_wav)

        if result.returncode != 0:
            stderr_msg = result.stderr.decode(errors='replace')[:300]
            log.error(f"[Live/{session.session_id}] slice extract failed: {stderr_msg}")
            return None

        session._prev_wav_duration = total_dur
        session.chunk_index += 1
        session.chunk_paths.append(wav_path)

        # Update concat list immediately for visibility
        _write_concat_list(session)

        return wav_path
    except Exception as exc:
        log.error(f"[Live/{session.session_id}] chunk save failed: {exc}")
        return None


def _write_concat_list(session: LiveSession) -> str:
    """Helper to write the ffmpeg concat list."""
    list_path = os.path.join(session.audio_dir, "concat_list.txt")
    with open(list_path, "w") as fh:
        for cp in session.chunk_paths:
            # use forward slashes for ffmpeg path safety
            safe_path = os.path.abspath(cp).replace("\\", "/")
            fh.write(f"file '{safe_path}'\n")
    return list_path


def _merge_chunks_to_wav(session: LiveSession) -> Optional[str]:
    """
    Concatenate all accumulated WAV chunks into a 16 kHz mono WAV via ffmpeg.
    Returns the WAV path or None on failure.
    """
    if not session.chunk_paths:
        return None

    list_path = _write_concat_list(session)
    out_path  = os.path.join(session.audio_dir, "merged.wav")

    try:

        # Since they are all WAV now, concat is fast
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                out_path,
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            log.error(
                f"[Live/{session.session_id}] ffmpeg merge error: "
                f"{result.stderr.decode(errors='replace')[:300]}"
            )
            return None

        session.merged_path = out_path
        return out_path
    except Exception as exc:
        log.error(f"[Live/{session.session_id}] merge error: {exc}")
        return None


async def _transcribe_pass(
    session: LiveSession,
    loop: asyncio.AbstractEventLoop,
    incremental: bool = False,
) -> List[Dict]:
    """
    Run Whisper on audio chunks.
    If incremental is True, only transcribes the latest chunk.
    """
    from academic_system.whisper_transcriber1 import transcribe as whisper_transcribe

    if incremental:
        if not session.chunk_paths:
            return []
        
        target_idx = len(session.chunk_paths) - 1
        if target_idx <= session.last_transcribed_idx:
            return []
            
        target_chunk = session.chunk_paths[target_idx]
        offset = target_idx * LIVE_CHUNK_INTERVAL_SEC
        
        try:
            result = await loop.run_in_executor(
                None,
                lambda: whisper_transcribe(target_chunk, language=None, model_size=LIVE_WHISPER_MODEL),
            )
            session.last_transcribed_idx = target_idx
            
            # Apply offset to segments
            segments = result.get("segments", [])
            for s in segments:
                s["start"] += offset
                s["end"]   += offset
            return segments
        except Exception as exc:
            log.error(f"[Live/{session.session_id}] Whisper incremental error: {exc}")
            return []
    else:
        # Full pass
        merged = await loop.run_in_executor(None, _merge_chunks_to_wav, session)
        if not merged or not os.path.isfile(merged):
            return []

        try:
            result = await loop.run_in_executor(
                None,
                lambda: whisper_transcribe(merged, language=None, model_size=LIVE_WHISPER_MODEL),
            )
            return result.get("segments", [])
        except Exception as exc:
            log.error(f"[Live/{session.session_id}] Whisper full error: {exc}")
            return []


def _norm_seg(raw: Any) -> Dict[str, Any]:
    """Normalise a faster-whisper segment (NamedTuple, dict, or list) to our format."""
    try:
        if isinstance(raw, dict):
            return {
                "id":    raw.get("id", 0),
                "start": round(float(raw.get("start", 0)), 3),
                "end":   round(float(raw.get("end",   0)), 3),
                "text":  (raw.get("text", "") or "").strip(),
            }
        if hasattr(raw, "start"):
            return {
                "id":    getattr(raw, "id", 0),
                "start": round(float(raw.start), 3),
                "end":   round(float(raw.end),   3),
                "text":  (raw.text or "").strip(),
            }
        if isinstance(raw, (list, tuple)) and len(raw) >= 3:
            # Handle [start, end, text] format if encountered
            return {
                "id":    0,
                "start": round(float(raw[0]), 3),
                "end":   round(float(raw[1]), 3),
                "text":  str(raw[2]).strip(),
            }
    except Exception:
        pass
    return {"id": 0, "start": 0, "end": 0, "text": str(raw).strip()}


# ─────────────────────────────────────────────────────────────────────────────
#  REST endpoints
# ─────────────────────────────────────────────────────────────────────────────

class StartLectureRequest(BaseModel):
    title: str = "Live Lecture"


@router.post("/start", summary="Create a live lecture session")
async def start_live_lecture(
    body:         StartLectureRequest,
    current_user: auth.LocalUser = Depends(auth.get_current_user),
) -> JSONResponse:
    session_id = str(uuid.uuid4())
    session    = LiveSession(
        session_id = session_id,
        user_id    = current_user.id,
        title      = (body.title or "Live Lecture").strip(),
    )
    _live_sessions[session_id] = session
    
    # ── Create LiveLecture DB row ─────────────────────────────────────────────
    from database_v2 import SessionLocal, LiveLecture
    db = SessionLocal()
    try:
        new_live = LiveLecture(
            id            = uuid.UUID(session_id),
            user_id       = current_user.id,
            session_id    = session_id,
            title         = session.title,
            state         = "created",
            started_at    = session.started_at,
            pipeline_stem = session.pipeline_stem,
            full_text     = "",
        )
        db.add(new_live)
        db.commit()
    except Exception as exc:
        log.error(f"[Live DB] Failed to create DB row: {exc}")
        db.rollback()
    finally:
        db.close()
        
    log.info(f"[Live] Session created: {session_id}  user={current_user.username}")

    return JSONResponse({
        "session_id":    session_id,
        "state":         "created",
        "pipeline_stem": session.pipeline_stem,
        "next_steps": {
            "1_ws":       f"WS  /live/lecture/{session_id}?token=<JWT>",
            "2_end":      f"POST /live/end/{session_id}",
            "3_status":   f"GET  /live/status/{session_id}",
            "4_results":  f"GET  /results/notes/{session.pipeline_stem}  (after pipeline completes)",
        },
    })


@router.post("/end/{session_id}", summary="End recording and trigger the academic pipeline")
async def end_live_lecture(
    session_id:   str,
    current_user: auth.LocalUser = Depends(auth.get_current_user),
) -> JSONResponse:
    session = _live_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    if session.user_id != current_user.id:
        raise HTTPException(403, "Not your session.")
    if session.state not in ("created", "recording", "stopped"):
        return JSONResponse({
            "session_id": session_id,
            "state":      session.state,
            "message":    f"Session already in state '{session.state}'.",
        })

    session.state    = "processing"
    if not session.ended_at:
        session.ended_at = time.time()
    log.info(f"[Live] Ending session {session_id} — {len(session.segments)} segments.")

    asyncio.create_task(_finish_session(session, current_user.id))

    return JSONResponse({
        "session_id":    session_id,
        "state":         "processing",
        "segment_count": len(session.segments),
        "word_count":    len(session.full_text.split()),
        "pipeline_stem": session.pipeline_stem,
        "poll":          f"GET /live/status/{session_id}",
        "results": {
            "notes":          f"GET  /results/notes/{session.pipeline_stem}",
            "pdf":            f"GET  /results/pdf/{session.pipeline_stem}",
            "flashcards_gen": f"POST /generate/flashcards/{session.pipeline_stem}",
            "flashcards":     f"GET  /results/flashcards/{session.pipeline_stem}",
            "quiz":           f"GET  /results/quiz/{session.pipeline_stem}",
            "graph":          f"GET  /results/graph/{session.pipeline_stem}",
        },
    })


@router.post("/cancel/{session_id}", summary="Cancel and discard a live lecture session")
async def cancel_live_lecture(
    session_id:   str,
    current_user: auth.LocalUser = Depends(auth.get_current_user),
) -> JSONResponse:
    """
    Discard a live lecture session entirely:
      1. Remove audio files from disk
      2. Delete DB rows (LiveLecture + LiveTranscriptionSegment)
      3. Remove in-memory session
    Only allowed when the session is in 'created', 'recording', or 'stopped' state.
    """
    session = _live_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    if session.user_id != current_user.id:
        raise HTTPException(403, "Not your session.")
    if session.state in ("processing", "done"):
        return JSONResponse({
            "session_id": session_id,
            "state":      session.state,
            "message":    f"Cannot cancel — session is already '{session.state}'.",
        }, status_code=409)

    log.info(f"[Live] Cancelling session {session_id} — discarding all data.")

    # 1. Remove audio files from disk
    try:
        if os.path.isdir(session.audio_dir):
            shutil.rmtree(session.audio_dir, ignore_errors=True)
            log.info(f"[Live] Removed audio dir: {session.audio_dir}")
    except Exception as exc:
        log.error(f"[Live] Failed to remove audio dir: {exc}")

    # 2. Delete DB rows
    try:
        from database_v2 import SessionLocal, LiveLecture, LiveTranscriptionSegment
        db = SessionLocal()
        try:
            db_live = db.query(LiveLecture).filter(
                LiveLecture.session_id == session_id
            ).first()
            if db_live:
                db.query(LiveTranscriptionSegment).filter(
                    LiveTranscriptionSegment.live_lecture_id == db_live.id
                ).delete()
                db.delete(db_live)
                db.commit()
                log.info(f"[Live] DB rows deleted for session {session_id}")
        except Exception as exc:
            log.error(f"[Live DB] Cancel cleanup failed: {exc}")
            db.rollback()
        finally:
            db.close()
    except Exception as exc:
        log.error(f"[Live DB] Import/cleanup error: {exc}")

    # 3. Remove from in-memory registry
    _live_sessions.pop(session_id, None)

    return JSONResponse({
        "session_id": session_id,
        "state":      "cancelled",
        "message":    "Session cancelled and all data discarded.",
    })


@router.get("/status/{session_id}", summary="Poll session state and pipeline progress")
async def live_status(
    session_id:   str,
    current_user: auth.LocalUser = Depends(auth.get_current_user),
) -> JSONResponse:
    session = _live_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    if session.user_id != current_user.id:
        raise HTTPException(403, "Not your session.")

    prog, status = 0, ""
    try:
        main_mod = _get_main_mod()
        prog     = main_mod.pipeline_progress
        status   = main_mod.pipeline_status
    except Exception:
        pass

    payload = {
        **session.to_dict(),
        "pipeline_progress": prog   if session.pipeline_started else 0,
        "pipeline_status":   status if session.pipeline_started else "",
        "results_ready":     session.state == "done",
    }
    if session.state == "done":
        payload["results"] = {
            "notes":          f"GET  /results/notes/{session.pipeline_stem}",
            "pdf":            f"GET  /results/pdf/{session.pipeline_stem}",
            "flashcards_gen": f"POST /generate/flashcards/{session.pipeline_stem}",
            "quiz":           f"GET  /results/quiz/{session.pipeline_stem}",
            "graph":          f"GET  /results/graph/{session.pipeline_stem}",
        }
    return JSONResponse(payload)


@router.get("/transcript/{session_id}", summary="Full transcript (live or final)")
async def live_transcript(
    session_id:   str,
    current_user: auth.LocalUser = Depends(auth.get_current_user),
) -> JSONResponse:
    session = _live_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    if session.user_id != current_user.id:
        raise HTTPException(403, "Not your session.")

    return JSONResponse({
        "session_id":    session_id,
        "state":         session.state,
        "full_text":     session.full_text,
        "segments":      session.segments,
        "segment_count": len(session.segments),
        "word_count":    len(session.full_text.split()),
    })


@router.get("/sessions", summary="List your live lecture sessions")
async def list_live_sessions(
    current_user: auth.LocalUser = Depends(auth.get_current_user),
) -> JSONResponse:
    sessions = sorted(
        [s.to_dict() for s in _live_sessions.values() if s.user_id == current_user.id],
        key=lambda s: s["started_at"],
        reverse=True,
    )
    return JSONResponse({"sessions": sessions, "total": len(sessions)})


# ─────────────────────────────────────────────────────────────────────────────
#  WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/lecture/{session_id}")
async def live_lecture_ws(websocket: WebSocket, session_id: str) -> None:
    """
    Binary WebSocket for real-time audio streaming.

    CLIENT → SERVER
    ───────────────
      Binary frames  — raw audio (WebM/Opus from MediaRecorder.ondataavailable)
      Text frame     — JSON control messages:
                         {"type": "ping"}
                         {"type": "end"}   ← graceful close (same as POST /live/end)

    SERVER → CLIENT  (JSON text frames)
    ───────────────
      {"type": "connected",   "session_id": "...", "state": "recording"}
      {"type": "segment",     "id": 3, "text": "Hello world", "start": 0.0, "end": 1.4}
      {"type": "heartbeat",   "segment_count": 12, "word_count": 87}
      {"type": "error",       "message": "..."}
      {"type": "done",        "pipeline_stem": "...", "segment_count": 45}

    Authentication
    ──────────────
      Pass JWT as ?token=<JWT> query param.
      Browsers cannot set Authorization headers on WebSocket connections.
    """
    token        = websocket.query_params.get("token", "")
    current_user = None
    if token:
        try:
            current_user = auth.decode_access_token(token)
        except Exception:
            pass

    await websocket.accept()

    async def _send(data: dict) -> None:
        try:
            await websocket.send_text(json.dumps(data))
        except Exception:
            pass

    if current_user is None:
        await _send({"type": "error", "message": "Unauthorized — pass ?token=<JWT>"})
        await websocket.close(code=4001)
        return

    session = _live_sessions.get(session_id)
    if session is None:
        await _send({"type": "error", "message": "Session not found — call POST /live/start first"})
        await websocket.close(code=4004)
        return

    if session.user_id != current_user.id:
        await _send({"type": "error", "message": "Forbidden"})
        await websocket.close(code=4003)
        return

    session.state = "recording"
    await _send({"type": "connected", "session_id": session_id, "state": "recording"})
    log.info(f"[Live WS] Connected: {session_id}  user={current_user.username}")

    loop                 = asyncio.get_event_loop()
    pending_bytes        = bytearray()
    last_transcribe_time = time.monotonic()
    hb_tick              = 0

    async def _do_transcribe() -> None:
        nonlocal last_transcribe_time
        last_transcribe_time = time.monotonic()
        prev_count = len(session.segments)
        
        # Transcribe the latest chunk
        raw_segs = await _transcribe_pass(session, loop, incremental=True)
        
        if raw_segs:
            # Normalize all segments once
            normed_chunk_segs = [_norm_seg(s) for s in raw_segs]
            chunk_text = " ".join(s["text"] for s in normed_chunk_segs).strip()

            # ── 1. Save standalone JSON for this chunk ──
            try:
                chunk_idx = session.last_transcribed_idx
                json_path = os.path.join(session.audio_dir, f"chunk_{chunk_idx:05d}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "chunk_index": chunk_idx,
                        "timestamp":   time.time(),
                        "text":        chunk_text,
                        "segments":    normed_chunk_segs
                    }, f, indent=2, ensure_ascii=False)
            except Exception as e:
                log.error(f"[Live WS] Failed to save chunk JSON: {e}")

            # ── 2. Persist to DB immediately ──
            from database_v2 import SessionLocal, LiveTranscriptionSegment
            db = SessionLocal()
            try:
                lecture_id = uuid.UUID(session_id)
                for raw in raw_segs:
                    seg = _norm_seg(raw)
                    if session.add_segment(seg):
                        # Send to UI
                        await _send({"type": "segment", **seg})
                        
                        # Save to DB
                        db_seg = LiveTranscriptionSegment(
                            id              = uuid.uuid4(),
                            live_lecture_id = lecture_id,
                            segment_index   = len(session.segments) - 1,
                            start_time      = float(seg["start"]),
                            end_time        = float(seg["end"]),
                            text            = seg["text"]
                        )
                        db.add(db_seg)
                db.commit()
            except Exception as e:
                log.error(f"[Live DB] Real-time segment save failed: {e}")
                db.rollback()
            finally:
                db.close()

        added = len(session.segments) - prev_count
        if added:
            log.debug(f"[Live WS] {session_id}: +{added} new segments ({len(session.segments)} total)")

    try:
        while True:
            # 2-second receive timeout to allow periodic heartbeat + transcription
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=2.0)
            except asyncio.TimeoutError:
                hb_tick += 1
                if hb_tick % 5 == 0:
                    await _send({
                        "type":          "heartbeat",
                        "segment_count": len(session.segments),
                        "word_count":    len(session.full_text.split()),
                    })
                elapsed = time.monotonic() - last_transcribe_time
                if elapsed >= LIVE_CHUNK_INTERVAL_SEC and session.chunk_paths:
                    await _do_transcribe()
                continue

            msg_type = msg.get("type", "")

            if msg_type == "websocket.disconnect":
                log.info(f"[Live WS] Disconnect: {session_id}")
                break

            if msg_type != "websocket.receive":
                continue

            # Binary = audio chunk
            if msg.get("bytes"):
                pending_bytes.extend(msg["bytes"])
                if len(pending_bytes) >= LIVE_MIN_CHUNK_BYTES:
                    _save_chunk(session, bytes(pending_bytes))
                    pending_bytes.clear()
                elapsed = time.monotonic() - last_transcribe_time
                if elapsed >= LIVE_CHUNK_INTERVAL_SEC and session.chunk_paths:
                    await _do_transcribe()

            # Text = control message
            elif msg.get("text"):
                try:
                    ctrl = json.loads(msg["text"])
                except json.JSONDecodeError:
                    ctrl = {}
                if ctrl.get("type") == "ping":
                    await _send({"type": "pong"})
                elif ctrl.get("type") in ("end", "stop"):
                    # 'stop' = recording stopped, user hasn't decided yet
                    # 'end'  = legacy compat, same behavior
                    if pending_bytes:
                        _save_chunk(session, bytes(pending_bytes))
                        pending_bytes.clear()
                    await _do_transcribe()
                    log.info(f"[Live WS] Client sent '{ctrl.get('type')}': {session_id}")
                    break

    except WebSocketDisconnect:
        log.info(f"[Live WS] WebSocketDisconnect: {session_id}")
    except Exception as exc:
        log.error(f"[Live WS] Error for {session_id}: {exc}", exc_info=True)
        await _send({"type": "error", "message": str(exc)})
    finally:
        # Flush remaining buffered bytes
        if pending_bytes:
            _save_chunk(session, bytes(pending_bytes))

        # Final transcription pass
        try:
            await _do_transcribe()
        except Exception:
            pass

        # Move to 'stopped' state — waiting for user to decide (process or cancel)
        # Do NOT auto-trigger the pipeline; POST /live/end does that explicitly.
        if session.state == "recording":
            session.state    = "stopped"
            session.ended_at = time.time()

        await _send({
            "type":          "stopped",
            "pipeline_stem": session.pipeline_stem,
            "segment_count": len(session.segments),
        })
        try:
            await websocket.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Post-session pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _get_main_mod() -> Any:
    """Lazy-import the main application module (handles both naming conventions)."""
    for name in ("main_v7_9", "main_v7-9"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue
    raise ImportError("Cannot find main app module (tried main_v7_9 and main_v7-9).")


async def _finish_session(session: LiveSession, user_id: int) -> None:
    """
    Called when recording ends (POST /live/end or WS disconnect).
    Runs the final Whisper pass then Phase 2 + Phase 3 of the academic pipeline.
    """
    log.info(f"[Live] _finish_session: {session.session_id}")
    loop = asyncio.get_event_loop()

    try:
        # Final Whisper pass (full merge for accuracy)
        final_segs = await _transcribe_pass(session, loop, incremental=False)
        for raw in final_segs:
            session.add_segment(_norm_seg(raw))

        transcript_result = {
            "text":     session.full_text,
            "segments": session.segments,
            "language": "en",
        }

        main_mod  = _get_main_mod()
        fake_path = session.pipeline_stem
        stem      = session.pipeline_stem

        # Seed academic_results so all GET /results/* endpoints resolve immediately
        main_mod.academic_results[fake_path] = {
            "input_type":            "live",
            "video_path":            fake_path,
            "live_session_id":       session.session_id,
            "live_title":            session.title,
            "frames_index":          [],
            "per_frame_details":     [],
            "audio_analysis":        transcript_result,
            "audio_topics":          {},
            "lecture_summary":       {},
            "detected_language":     {"code": "en", "name": "English", "rtl": False},
            "deduped_concepts":      [],
            "deduped_formulas":      [],
            "study_notes":           None,
            "flashcards":            [],
            "quiz":                  [],
            "knowledge_graph":       None,
            "pdf_report_path":       None,
            "slide_change_stats":    {},
            "duration_sec":          (session.ended_at or time.time()) - session.started_at,
            "whisper_model":         LIVE_WHISPER_MODEL,
            "user_id":               user_id,
            "created_at":            session.started_at,
        }
        main_mod._flashcard_states[stem] = {
            "state": "idle", "flashcard_count": 0, "quiz_count": 0, "error": None,
        }

        session.pipeline_started = True
        session.state            = "processing"

        await _run_phase2_for_live(
            fake_path         = fake_path,
            stem              = stem,
            transcript_result = transcript_result,
            user_id           = user_id,
            session_title     = session.title,
            main_mod          = main_mod,
        )

        session.state = "done"
        
        # ── Update LiveLecture DB row ─────────────────────────────────────────
        from database_v2 import SessionLocal, LiveLecture, LiveTranscriptionSegment
        db = SessionLocal()
        try:
            db_live = db.query(LiveLecture).filter(LiveLecture.session_id == session.session_id).first()
            if db_live:
                db_live.state = "done"
                db_live.ended_at = session.ended_at
                db_live.full_text = session.full_text
                
                # Clear real-time segments, replace with final accurate set
                db.query(LiveTranscriptionSegment).filter(
                    LiveTranscriptionSegment.live_lecture_id == db_live.id
                ).delete()
                
                for i, seg in enumerate(session.segments):
                    db_seg = LiveTranscriptionSegment(
                        id=uuid.uuid4(),
                        live_lecture_id=db_live.id,
                        segment_index=i,
                        start_time=float(seg.get("start", 0)),
                        end_time=float(seg.get("end", 0)),
                        text=seg.get("text", "")
                    )
                    db.add(db_seg)
                db.commit()
        except Exception as exc:
            log.error(f"[Live DB] Failed to update DB on finish: {exc}")
            db.rollback()
        finally:
            db.close()
            
        log.info(f"[Live] Pipeline complete: {session.session_id}")

    except Exception as exc:
        log.error(f"[Live] _finish_session failed: {exc}", exc_info=True)
        session.state          = "error"
        session.pipeline_error = str(exc)
        
        from database_v2 import SessionLocal, LiveLecture
        db = SessionLocal()
        try:
            db_live = db.query(LiveLecture).filter(LiveLecture.session_id == session.session_id).first()
            if db_live:
                db_live.state = "error"
                db_live.ended_at = time.time()
                db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()


async def _run_phase2_for_live(
    fake_path:         str,
    stem:              str,
    transcript_result: Dict[str, Any],
    user_id:           int,
    session_title:     str,
    main_mod:          Any,
) -> None:
    """
    Phase 2 + Phase 3 for a live lecture.

    Identical logic to the video pipeline inner loop in run_academic_pipeline(),
    but skips Phase 1 (no frames, no Whisper re-run — transcript already available).

    Call 1  — prompt_metadata       → lecture title, topics, key concepts
    Call 2  — prompt_study_notes_text → full markdown study notes
    Phase 3 — knowledge graph + PDF report (parallel threads)
    DB      — save_pipeline_result_to_db
    """
    from video_pipeline.config1 import config
    from video_pipeline.reasoning.phi3_engine import Phi3Reasoner as LlamaReasoner
    from video_pipeline.utils.device import setup_device
    from academic_system.pdf_generator import generate_pdf_report
    from academic_system.prompts1 import prompt_metadata, prompt_study_notes_text

    vr = main_mod.academic_results[fake_path]

    # ── Reuse shared LLM (avoid reloading Phi-3 if already in VRAM) ──────────
    llm = main_mod._shared_llm
    if llm is None and main_mod._shared_llm_ref and main_mod._shared_llm_ref[0]:
        llm = main_mod._shared_llm_ref[0]
    if llm is None:
        log.info(f"[Live Phase2/{stem}] Loading Phi-3 LLM…")
        llm = LlamaReasoner(
            model_id       = config.reasoning_model_id,
            max_new_tokens = config.max_reasoning_tokens,
            device         = setup_device(),
            load_in_4bit   = config.phi3_load_in_4bit,
            adapter_path   = config.phi3_adapter_path or None,
        )
        main_mod._shared_llm = llm

    transcript = transcript_result.get("text", "").strip()
    words = transcript.split()
    
    if len(words) < 5:
        log.warning(f"[Live Phase2/{stem}] Transcript too short ({len(words)} words). Skipping LLM calls to prevent hallucination.")
        meta = {
            "lecture_title": session_title or stem,
            "subject_area": "Unknown",
            "topics": ["No topics captured"],
            "learning_outcomes": [],
            "summary": "No meaningful audio information was captured during the live session.",
            "difficulty": "Unknown",
            "key_concepts": []
        }
        
        lecture_title   = meta["lecture_title"]
        lecture_summary = {
            "lecture_title":     lecture_title,
            "subject_area":      meta["subject_area"],
            "main_topics":       meta["topics"],
            "learning_outcomes": meta["learning_outcomes"],
            "summary":           meta["summary"],
            "difficulty_level":  meta["difficulty"],
        }
        audio_topics = {
            "lecture_title":    lecture_title,
            "subject_area":     meta["subject_area"],
            "topics_covered":   meta["topics"],
            "key_concepts":     [],
            "important_points": meta["learning_outcomes"],
            "summary":          meta["summary"],
        }
        notes_md = f"# Study Notes: {lecture_title}\n\nNo meaningful audio information was captured during the live session."
    else:
        lang_info  = main_mod._lang_detector.from_code("en")
        _patch     = lambda p: main_mod._lang_detector.patch_prompt(p, lang_info)

        # ── Call 1: metadata ──────────────────────────────────────────────────────
        log.info(f"[Live Phase2/{stem}] Call 1 — metadata…")
        meta: Dict[str, Any] = {}
        try:
            c1 = _patch(
                prompt_metadata(fake_path, [], transcript, sample_n=0, max_concepts=6)
            )
            meta = main_mod.serialize(main_mod._llm_reason(llm, c1, main_mod._MAX_TOKENS_META)) or {}
            if not meta:
                raw_out = getattr(llm, "_last_raw_output", "") or ""
                if raw_out:
                    meta = main_mod._extract_first_json_object(raw_out) or {}
        except Exception as exc:
            log.error(f"[Live Phase2/{stem}] Call 1 failed: {exc}", exc_info=True)

        lecture_title   = meta.get("lecture_title") or session_title or stem
        lecture_summary = {
            "lecture_title":     lecture_title,
            "subject_area":      meta.get("subject_area", ""),
            "main_topics":       meta.get("topics", []),
            "learning_outcomes": meta.get("learning_outcomes", []),
            "summary":           meta.get("summary", ""),
            "difficulty_level":  meta.get("difficulty", ""),
        }
        audio_topics = {
            "lecture_title":    lecture_title,
            "subject_area":     meta.get("subject_area", ""),
            "topics_covered":   meta.get("topics", []),
            "key_concepts":     [{"concept": c, "explanation": ""} for c in meta.get("key_concepts", [])],
            "important_points": meta.get("learning_outcomes", []),
            "summary":          meta.get("summary", ""),
        }

        # ── Call 2: study notes ───────────────────────────────────────────────────
        log.info(f"[Live Phase2/{stem}] Call 2 — study notes…")
        notes_md = ""
        try:
            c2 = _patch(
                prompt_study_notes_text(
                    lecture_title     = lecture_title,
                    subject_area      = meta.get("subject_area", "General"),
                    difficulty        = meta.get("difficulty", ""),
                    topics            = meta.get("topics", []),
                    key_concepts      = meta.get("key_concepts", []),
                    learning_outcomes = meta.get("learning_outcomes", []),
                    summary           = meta.get("summary", ""),
                    formulas          = [],
                )
            )
            notes_md = main_mod._llm_reason_text(llm, c2, main_mod._MAX_TOKENS_NOTES)
        except Exception as exc:
            log.error(f"[Live Phase2/{stem}] Call 2 failed: {exc}", exc_info=True)

        # Fallback notes if LLM returns nothing
        if not (notes_md or "").strip():
            lines = [f"# Study Notes: {lecture_title}", ""]
            if meta.get("summary"):
                lines += ["## Overview", "", meta["summary"], ""]
            for t in meta.get("topics", []):
                lines.append(f"- {t}")
            notes_md = "\n".join(lines)
        elif not notes_md.lstrip().startswith("#"):
            notes_md = f"# Study Notes: {lecture_title}\n\n{notes_md}"

    notes_path = main_mod.write_text_file(
        os.path.join(main_mod.NOTES_DIR, f"{stem}_study_notes.md"), notes_md
    )

    # ── Deduplication (no frames → quick pass) ────────────────────────────────
    deduped_concepts: List[str] = []
    deduped_formulas: List[str] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fc = pool.submit(main_mod._deduplicator.deduplicate_concepts, [])
            ff = pool.submit(main_mod._deduplicator.deduplicate_formulas,  [])
            deduped_concepts = fc.result()
            deduped_formulas = ff.result()
    except Exception:
        pass

    # ── Persist in-memory ─────────────────────────────────────────────────────
    vr.update({
        "audio_topics":          audio_topics,
        "lecture_summary":       lecture_summary,
        "total_frames_analysed": 0,
        "study_notes":           notes_md,
        "study_notes_path":      notes_path,
        "deduped_concepts":      deduped_concepts,
        "deduped_formulas":      deduped_formulas,
        "flashcards":            [],
        "quiz":                  [],
    })
    main_mod._flashcard_states[stem] = {
        "state": "idle", "flashcard_count": 0, "quiz_count": 0, "error": None,
    }

    # ── v3.3.1: Save standalone transcript files (match video_transcription tool) ──
    try:
        out_dir = os.path.join("video_transcription", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        
        txt_out = os.path.join(out_dir, f"{stem}_transcript.txt")
        json_out = os.path.join(out_dir, f"{stem}_transcript.json")
        
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(transcript)
            
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump({
                "language": transcript_result.get("language", "auto"),
                "text": transcript,
                "segments": transcript_result.get("segments", [])
            }, f, indent=2, ensure_ascii=False)
            
        log.info(f"[Live Phase2/{stem}] Standalone transcripts saved to {out_dir}")
    except Exception as exc:
        log.warning(f"[Live Phase2/{stem}] Failed to save standalone transcripts: {exc}")

    # ── Phase 3: knowledge graph + PDF (parallel threads) ─────────────────────
    def _build_graph() -> None:
        try:
            graph = main_mod._graph_builder.build([], audio_topics, lecture_summary)
            d3    = main_mod._graph_builder.to_d3_json(graph)
            gpath = main_mod._graph_builder.save(
                graph,
                os.path.join(main_mod.GRAPH_DIR, f"{stem}_knowledge_graph.json"),
            )
            vr["knowledge_graph"]      = d3
            vr["knowledge_graph_path"] = gpath
            log.info(f"[Live Phase2/{stem}] Knowledge graph built.")
        except Exception as exc:
            log.error(f"[Live Phase2/{stem}] Graph failed: {exc}")

    def _build_pdf() -> None:
        try:
            ppath = generate_pdf_report(
                video_path      = fake_path,
                pdf_dir         = main_mod.PDF_DIR,
                lecture_summary = lecture_summary,
                audio_topics    = audio_topics,
                frame_analyses  = [],
                flashcards      = [],
                transcript_text = transcript,
            )
            vr["pdf_report_path"] = ppath
            log.info(f"[Live Phase2/{stem}] PDF: {ppath}")
        except Exception as exc:
            log.error(f"[Live Phase2/{stem}] PDF failed: {exc}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        gf = pool.submit(_build_graph)
        pf = pool.submit(_build_pdf)
        gf.result()
        pf.result()

    # ── DB persistence ────────────────────────────────────────────────────────
    if user_id:
        try:
            from db_actions import save_pipeline_result_to_db
            save_pipeline_result_to_db(user_id, fake_path, vr, stem)
            log.info(f"[Live Phase2/{stem}] Persisted to DB.")
        except Exception as exc:
            log.error(f"[Live Phase2/{stem}] DB persist failed (non-fatal): {exc}")