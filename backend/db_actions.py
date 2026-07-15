"""
AcademIQ — DB Actions v1.1
===========================
Helpers called from the pipeline to persist results to PostgreSQL.

Fixes in v1.1
-------------
- save_flashcards_to_db: now called even when lists are empty (idempotent upsert)
- save_pipeline_result_to_db: persists media + video/image/document + notes rows
- get_db_media_states: replaces _collect_disk_lecture_stems for DB-backed /status
- get_db_user_progress_stats: for /dashboard/stats engagement block
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Internal import helper (avoids circular imports at module load time)
# ──────────────────────────────────────────────────────────────────────────────

def _db():
    """Return a fresh SQLAlchemy session. Caller must close it."""
    from database_v2 import SessionLocal
    return SessionLocal()


def _models():
    from database_v2 import User, Media, Video, Image, Document, Flashcard, QuizQuestion, Note, Progress, TranscriptionSegment, MediaResultStats
    return User, Media, Video, Image, Document, Flashcard, QuizQuestion, Note, Progress, TranscriptionSegment, MediaResultStats


# ──────────────────────────────────────────────────────────────────────────────
#  FLASHCARD + QUIZ PERSIST
# ──────────────────────────────────────────────────────────────────────────────

def save_flashcards_to_db(
    user_id: str,
    video_stem: str,
    flashcards: List[Dict],
    quiz: List[Dict],
) -> None:
    """
    Persist generated flashcards and quiz questions to PostgreSQL.

    Called from _run_flashcard_generation() regardless of whether the
    lists are empty — this makes the function idempotent and ensures
    the media row is always located even when generation produced 0 items.
    """
    _, Media, _, _, _, Flashcard, QuizQuestion, _, _, _, _ = _models()
    db = _db()
    try:
        # ── Find the media row that owns this stem ────────────────────────────
        # Prioritise newest record if multiple exist
        media = (
            db.query(Media)
            .filter(Media.batch_stem == video_stem, Media.user_id == user_id)
            .order_by(Media.uploaded_at.desc())
            .first()
        )

        if media is None:
            # Try matching on storage_path stem as a fallback
            all_media = (
                db.query(Media)
                .filter(Media.user_id == user_id)
                .order_by(Media.uploaded_at.desc())
                .all()
            )
            for m in all_media:
                if Path(m.storage_path or "").stem == video_stem:
                    media = m
                    break

        if media is None:
            logger.error(
                f"[DB] save_flashcards_to_db: no Media row found for "
                f"stem='{video_stem}' user='{user_id}'. Skipping DB persist."
            )
            return

        media_id = media.id
        logger.info(
            f"[DB] Persisting {len(flashcards)} flashcards + "
            f"{len(quiz)} quiz questions for media_id={media_id}"
        )

        # ── Delete old rows so re-generation is clean ─────────────────────────
        db.query(Flashcard).filter(Flashcard.media_id == media_id).delete()
        db.query(QuizQuestion).filter(QuizQuestion.media_id == media_id).delete()

        # ── Insert flashcards ─────────────────────────────────────────────────
        for i, card in enumerate(flashcards):
            if not card.get("question"):
                continue
            db.add(Flashcard(
                id         = uuid.uuid4(),
                media_id   = media_id,
                card_index = i,
                question   = card.get("question", ""),
                answer     = card.get("answer", ""),
                topic      = card.get("topic") or None,
                difficulty = card.get("difficulty", "medium"),
            ))

        # ── Insert quiz questions ─────────────────────────────────────────────
        for i, q in enumerate(quiz):
            if not q.get("question"):
                continue
            # Normalise options: accept {"A":..,"B":..} or {"options":{"A":..}}
            opts = q.get("options") or {
                k: q[k] for k in ("A", "B", "C", "D") if k in q
            }
            db.add(QuizQuestion(
                id             = uuid.uuid4(),
                media_id       = media_id,
                question_index = i,
                question       = q.get("question", ""),
                options        = opts,
                correct_answer = q.get("correct_answer", ""),
                explanation    = q.get("explanation") or None,
                topic          = q.get("topic") or None,
            ))

        db.commit()
        logger.info(
            f"[DB] ✅ Saved {len(flashcards)} flashcards + "
            f"{len(quiz)} quiz questions for stem='{video_stem}'"
        )

    except Exception as exc:
        db.rollback()
        logger.error(
            f"[DB] save_flashcards_to_db FAILED for stem='{video_stem}': {exc}",
            exc_info=True,
        )
    finally:
        db.close()


# ──────────────────────────────────────────────────────────────────────────────
#  PIPELINE RESULT PERSIST  (called after Phase 2 completes)
# ──────────────────────────────────────────────────────────────────────────────

def save_pipeline_result_to_db(
    user_id: str,
    source_path: str,          # original local file path (video / first image / pdf)
    vr: Dict[str, Any],        # the academic_results[...] dict
    batch_stem: str,
) -> None:
    """
    Upsert Media + typed extension row + study notes into PostgreSQL
    after Phase 2 (or image/PDF pipeline) completes.
    """
    _, Media, Video, Image, Document, _, _, Note, _, TranscriptionSegment, MediaResultStats = _models()
    db = _db()
    try:
        input_type = vr.get("input_type", "video")  # "video" | "images" | "document"

        # ── Locate or create Media row ────────────────────────────────────────
        media = (
            db.query(Media)
            .filter(Media.batch_stem == batch_stem, Media.user_id == user_id)
            .order_by(Media.uploaded_at.desc())
            .first()
        )

        if media is None:
            # Fallback: find by storage_path stem
            all_media = db.query(Media).filter(Media.user_id == user_id).all()
            for m in all_media:
                if Path(m.storage_path or "").stem == batch_stem:
                    media = m
                    break

        if media is None:
            # Create it now — upload endpoint may have been skipped / restarted
            file_size = 0
            try:
                file_size = os.path.getsize(source_path) if os.path.isfile(source_path) else 0
            except OSError:
                pass

            media_type_val = (
                "image" if input_type == "images" else
                "document" if input_type in ("document", "pdf") else
                "video"
            )
            media = Media(
                id                = uuid.uuid4(),
                user_id           = user_id,
                media_type        = media_type_val,
                original_filename = Path(source_path).name,
                storage_path      = source_path,
                minio_object_key  = "",
                file_size_bytes   = file_size,
                batch_stem        = batch_stem,
            )
            db.add(media)
            db.flush()   # get media.id before inserting children
            logger.info(f"[DB] Created Media row id={media.id} stem='{batch_stem}'")
        else:
            logger.info(f"[DB] Found existing Media row id={media.id} stem='{batch_stem}'")

        media_id = media.id
        ls = vr.get("lecture_summary") or {}

        # ── Upsert typed extension row ─────────────────────────────────────────
        if input_type == "video":
            row = db.query(Video).filter(Video.media_id == media_id).first()
            if row is None:
                row = Video(id=uuid.uuid4(), media_id=media_id)
                db.add(row)
            row.duration_sec          = vr.get("duration_sec")
            row.adaptive_fps          = vr.get("adaptive_fps")
            row.whisper_model         = vr.get("whisper_model")
            row.detected_language     = (vr.get("detected_language") or {}).get("code")
            row.lecture_title         = ls.get("lecture_title")
            row.subject_area          = ls.get("subject_area")
            row.difficulty_level      = ls.get("difficulty_level")
            row.summary               = ls.get("summary")
            row.main_topics           = ls.get("main_topics") or []
            row.learning_outcomes     = ls.get("learning_outcomes") or []
            row.transcription         = vr.get("audio_analysis") or vr.get("transcription")
            row.study_notes_path      = vr.get("study_notes_path")
            row.pdf_report_path       = vr.get("pdf_report_path")
            row.knowledge_graph_path  = vr.get("knowledge_graph_path")
            row.processed_at          = datetime.utcnow()

        elif input_type == "images":
            # For image batches, update batch-level fields on existing Image rows
            # (Image rows should already exist from the upload → pipeline flow)
            existing_imgs = db.query(Image).filter(Image.media_id == media_id).all()
            for img_row in existing_imgs:
                img_row.lecture_title     = ls.get("lecture_title")
                img_row.subject_area      = ls.get("subject_area")
                img_row.study_notes_path  = vr.get("study_notes_path")
                img_row.pdf_report_path   = vr.get("pdf_report_path")
                img_row.processed_at      = datetime.utcnow()

            # If no image rows exist yet, insert one placeholder so FK is valid
            if not existing_imgs:
                db.add(Image(
                    id               = uuid.uuid4(),
                    media_id         = media_id,
                    batch_stem       = batch_stem,
                    batch_index      = 0,
                    lecture_title    = ls.get("lecture_title"),
                    subject_area     = ls.get("subject_area"),
                    study_notes_path = vr.get("study_notes_path"),
                    pdf_report_path  = vr.get("pdf_report_path"),
                ))

        elif input_type in ("document", "pdf"):
            row = db.query(Document).filter(Document.media_id == media_id).first()
            if row is None:
                row = Document(id=uuid.uuid4(), media_id=media_id)
                db.add(row)
            row.lecture_title         = ls.get("lecture_title")
            row.subject_area          = ls.get("subject_area")
            row.difficulty_level      = ls.get("difficulty_level")
            row.summary               = ls.get("summary")
            row.main_topics           = ls.get("main_topics") or []
            row.learning_outcomes     = ls.get("learning_outcomes") or []
            row.study_notes_path      = vr.get("study_notes_path")
            row.pdf_report_path       = vr.get("pdf_report_path")
            row.knowledge_graph_path  = vr.get("knowledge_graph_path")
            row.processed_at          = datetime.utcnow()

        # ── Upsert study notes ────────────────────────────────────────────────
        notes_text = vr.get("study_notes") or ""
        if notes_text:
            note_row = db.query(Note).filter(Note.media_id == media_id).first()
            if note_row is None:
                note_row = Note(id=uuid.uuid4(), media_id=media_id, content=notes_text)
                db.add(note_row)
            else:
                note_row.content    = notes_text
                note_row.updated_at = datetime.utcnow()

        # ── Upsert Transcription Segments ─────────────────────────────────────
        transcription_obj = vr.get("audio_analysis") or vr.get("transcription")
        if transcription_obj and isinstance(transcription_obj, dict):
            segments = transcription_obj.get("segments", [])
            if segments:
                # Delete old segments
                db.query(TranscriptionSegment).filter(TranscriptionSegment.media_id == media_id).delete()
                
                for i, seg in enumerate(segments):
                    db.add(TranscriptionSegment(
                        id            = uuid.uuid4(),
                        media_id      = media_id,
                        segment_index = i,
                        start_time    = float(seg.get("start", 0)),
                        end_time      = float(seg.get("end", 0)),
                        text          = seg.get("text", "").strip(),
                        confidence    = float(seg.get("confidence", 1.0)) if seg.get("confidence") is not None else None
                    ))

                # ── Upsert MediaResultStats ───────────────────────────────────
                full_text = transcription_obj.get("text", "")
                word_count = len(full_text.split())
                duration = segments[-1].get("end", 0) if segments else 0
                lang_code = transcription_obj.get("language")
                
                stats_row = db.query(MediaResultStats).filter(MediaResultStats.media_id == media_id).first()
                if stats_row is None:
                    stats_row = MediaResultStats(id=uuid.uuid4(), media_id=media_id)
                    db.add(stats_row)
                
                stats_row.duration_sec  = float(duration)
                stats_row.segment_count = len(segments)
                stats_row.word_count    = word_count
                stats_row.language_code = str(lang_code) if lang_code else None
                stats_row.processed_at  = datetime.utcnow()

        db.commit()
        logger.info(f"[DB] ✅ Pipeline result persisted for stem='{batch_stem}'")

    except Exception as exc:
        db.rollback()
        logger.error(
            f"[DB] save_pipeline_result_to_db FAILED for stem='{batch_stem}': {exc}",
            exc_info=True,
        )
    finally:
        db.close()


# ──────────────────────────────────────────────────────────────────────────────
#  READ HELPERS  (used by /status and /dashboard/stats)
# ──────────────────────────────────────────────────────────────────────────────

def get_db_media_states(user_id: str) -> Dict[str, Dict[str, Any]]:
    """
    Return a dict keyed by batch_stem with status dicts compatible with
    the /status endpoint's video_states format.
    Replaces _collect_disk_lecture_stems() + _disk_lecture_state() for DB-backed flow.
    """
    _, Media, Video, Image, Document, Flashcard, QuizQuestion, Note, _, _, _ = _models()
    db = _db()
    result: Dict[str, Dict[str, Any]] = {}
    try:
        if isinstance(user_id, str):
            try:
                user_uuid = uuid.UUID(user_id)
            except ValueError:
                user_uuid = user_id
        else:
            user_uuid = user_id

        all_media = (
            db.query(Media)
            .filter(Media.user_id == user_uuid)
            .order_by(Media.uploaded_at.desc())
            .all()
        )

        for m in all_media:
            stem = m.batch_stem or Path(m.storage_path or "").stem
            if not stem:
                continue
            
            # Since all_media is desc by uploaded_at, the first 'stem' we see is the newest.
            # Skip any older entries for the same stem to avoid overwriting results.
            if stem in result:
                continue

            fc_count = db.query(Flashcard).filter(Flashcard.media_id == m.id).count()
            qz_count = db.query(QuizQuestion).filter(QuizQuestion.media_id == m.id).count()
            has_notes = db.query(Note).filter(Note.media_id == m.id).count() > 0

            # Derive lecture title from extension table
            lecture_title = None
            subject_area  = None
            pdf_ready     = False
            graph_ready   = False

            if m.media_type == "video":
                v = db.query(Video).filter(Video.media_id == m.id).first()
                if v:
                    lecture_title = v.lecture_title
                    subject_area  = v.subject_area
                    pdf_ready     = bool(v.pdf_report_path)
                    graph_ready   = bool(v.knowledge_graph_path)

            elif m.media_type == "image":
                img = db.query(Image).filter(Image.media_id == m.id).first()
                if img:
                    lecture_title = img.lecture_title
                    subject_area  = img.subject_area
                    pdf_ready     = bool(img.pdf_report_path)

            elif m.media_type == "document":
                d = db.query(Document).filter(Document.media_id == m.id).first()
                if d:
                    lecture_title = d.lecture_title
                    subject_area  = d.subject_area
                    pdf_ready     = bool(d.pdf_report_path)
                    graph_ready   = bool(d.knowledge_graph_path)

            input_type_map = {"video": "video", "image": "images", "document": "document"}

            result[stem] = {
                "source":                      "db",
                "video_stem":                  stem,
                "display_name":                lecture_title or stem,
                "lecture_title":               lecture_title,
                "subject_area":                subject_area,
                "input_type":                  input_type_map.get(m.media_type, m.media_type),
                "created_at":                  m.uploaded_at.timestamp() if m.uploaded_at else 0,
                "audio_ready":                 m.media_type == "video",
                "summary_ready":               has_notes or pdf_ready,
                "study_notes_ready":           has_notes,
                "pdf_ready":                   pdf_ready,
                "graph_ready":                 graph_ready,
                "flashcards_ready":            fc_count > 0,
                "flashcard_count":             fc_count,
                "quiz_ready":                  qz_count > 0,
                "quiz_count":                  qz_count,
                "flashcards_generation_state": "done" if (fc_count > 0 or qz_count > 0) else "idle",
                "flashcard_generate_url":      f"POST /generate/flashcards/{stem}",
                "pipeline_error":              None,
                "difficulty":                  None,
            }

    except Exception as exc:
        logger.error(f"[DB] get_db_media_states FAILED: {exc}", exc_info=True)
    finally:
        db.close()

    return result


def get_db_user_progress_stats(user_id: str) -> Dict[str, Any]:
    """Returns engagement stats for /dashboard/stats."""
    _, _, _, _, _, Flashcard, QuizQuestion, _, Progress, _, _ = _models()
    db = _db()
    try:
        if isinstance(user_id, str):
            try:
                user_uuid = uuid.UUID(user_id)
            except ValueError:
                user_uuid = user_id
        else:
            user_uuid = user_id

        total_reviews  = db.query(Progress).filter(Progress.user_id == user_uuid).count()
        correct_count  = db.query(Progress).filter(
            Progress.user_id == user_uuid, Progress.correct == True
        ).count()
        fc_reviews  = db.query(Progress).filter(
            Progress.user_id == user_uuid, Progress.item_type == "flashcard"
        ).count()
        qz_reviews  = db.query(Progress).filter(
            Progress.user_id == user_uuid, Progress.item_type == "quiz_question"
        ).count()

        avg_confidence = None
        rows = db.query(Progress.confidence).filter(
            Progress.user_id == user_uuid,
            Progress.confidence != None,
        ).all()
        if rows:
            avg_confidence = round(sum(r[0] for r in rows) / len(rows), 2)

        return {
            "total_reviews":     total_reviews,
            "flashcard_reviews": fc_reviews,
            "quiz_reviews":      qz_reviews,
            "correct_count":     correct_count,
            "avg_confidence":    avg_confidence,
        }
    except Exception as exc:
        logger.error(f"[DB] get_db_user_progress_stats FAILED: {exc}", exc_info=True)
        return {}
    finally:
        db.close()