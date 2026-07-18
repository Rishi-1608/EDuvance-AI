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
from sqlalchemy import func, cast, Integer, desc

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Internal import helper (avoids circular imports at module load time)
# ──────────────────────────────────────────────────────────────────────────────

def _db():
    """Return a fresh SQLAlchemy session. Caller must close it."""
    from database_v2 import SessionLocal
    return SessionLocal()


def _models():
    from database_v2 import User, Media, Video, Image, Document, Flashcard, QuizQuestion, Note, Progress, TranscriptionSegment, MediaResultStats, QuizSession
    return User, Media, Video, Image, Document, Flashcard, QuizQuestion, Note, Progress, TranscriptionSegment, MediaResultStats, QuizSession


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
    _, Media, _, _, _, Flashcard, QuizQuestion, _, _, _, _, _ = _models()
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
    _, Media, Video, Image, Document, _, _, Note, _, TranscriptionSegment, MediaResultStats, _ = _models()
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
                "live" if input_type == "live" else
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
        if input_type in ("video", "live"):
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
    _, Media, Video, Image, Document, Flashcard, QuizQuestion, Note, _, _, _, _ = _models()
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

            elif m.media_type == "live":
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

            input_type_map = {
                "video": "video",
                "image": "images",
                "document": "document",
                "live": "live"
            }

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
    _, _, _, _, _, Flashcard, QuizQuestion, _, Progress, _, _, QuizSession = _models()
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

        # ── Quiz Accuracy (Hybrid: Sessions + Progress Aggregation) ────────
        _, Media, Video, Image, Document, _, _, _, Progress, _, _, QuizSession = _models()
        
        # Helper to get title for a media record
        def get_media_title(m_id):
            m = db.query(Media).filter(Media.id == m_id).first()
            if not m: return "Unknown"
            t = m.batch_stem or "Unknown"
            if m.media_type in ("video", "live"):
                v = db.query(Video).filter(Video.media_id == m.id).first()
                if v: t = v.lecture_title or t
            elif m.media_type == "image":
                img = db.query(Image).filter(Image.media_id == m.id).first()
                if img: t = img.lecture_title or t
            elif m.media_type == "document":
                doc = db.query(Document).filter(Document.media_id == m.id).first()
                if doc: t = doc.lecture_title or t
            return t

        history_map = {} # timestamp_iso -> {score, title}

        # 1. Real sessions
        real_sessions = (
            db.query(QuizSession)
            .filter(QuizSession.user_id == user_uuid)
            .order_by(QuizSession.completed_at.asc())
            .all()
        )
        for s in real_sessions:
            history_map[s.completed_at.isoformat()] = {
                "score": s.accuracy_score,
                "title": get_media_title(s.media_id)
            }

        # 2. Aggregated sessions from Progress table (legacy/individual)
        agg_progress = (
            db.query(
                Progress.media_id,
                Progress.session_id,
                func.max(Progress.reviewed_at).label("latest"),
                func.count(Progress.id).label("total"),
                func.sum(cast(Progress.correct, Integer)).label("corrects")
            )
            .filter(Progress.user_id == user_uuid, Progress.item_type == "quiz_question")
            .group_by(Progress.media_id, Progress.session_id)
            .all()
        )

        for m_id, s_id, latest, total, corrects in agg_progress:
            if total > 0:
                acc = (corrects / total) * 100
                dt = latest.isoformat()
                if dt not in history_map:
                    history_map[dt] = {
                        "score": acc,
                        "title": get_media_title(m_id)
                    }

        # Convert to sorted list
        quiz_history = []
        for dt in sorted(history_map.keys()):
            quiz_history.append({
                "score": round(history_map[dt]["score"], 1),
                "date": dt,
                "title": history_map[dt]["title"]
            })

        quiz_accuracy_pct = 0
        if quiz_history:
            avg_accuracy = sum(h["score"] for h in quiz_history) / len(quiz_history)
            quiz_accuracy_pct = round(avg_accuracy, 1)

        total_interactions = total_reviews + len(real_sessions)

        return {
            "total_reviews":     total_reviews,
            "flashcard_reviews": fc_reviews,
            "quiz_reviews":      qz_reviews,
            "correct_count":     correct_count,
            "avg_confidence":    avg_confidence,
            "avg_confidence_pct": round((avg_confidence / 5.0) * 100, 1) if avg_confidence else 0,
            "quiz_accuracy_pct": quiz_accuracy_pct,
            "quiz_history":      quiz_history,
            "total_interactions": total_interactions,
            "flashcards_reviewed": fc_reviews,
            "total_quizzes":     len(quiz_history),
        }
    except Exception as exc:
        logger.error(f"[DB] get_db_user_progress_stats FAILED: {exc}", exc_info=True)
        return {}
    finally:
        db.close()


def save_quiz_session_to_db(
    user_id: str,
    video_stem: str,
    total_questions: int,
    correct_answers: int,
) -> None:
    """Persist a completed quiz session summary to the database."""
    _, Media, _, _, _, _, _, _, _, _, _, QuizSession = _models()
    db = _db()
    try:
        # Find media
        media = (
            db.query(Media)
            .filter(Media.batch_stem == video_stem, Media.user_id == user_id)
            .order_by(Media.uploaded_at.desc())
            .first()
        )
        
        if not media:
            logger.error(f"[DB] save_quiz_session: No media found for stem {video_stem}")
            return

        accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0
        
        session = QuizSession(
            id=uuid.uuid4(),
            user_id=user_id,
            media_id=media.id,
            total_questions=total_questions,
            correct_answers=correct_answers,
            accuracy_score=accuracy,
            completed_at=datetime.utcnow()
        )
        db.add(session)
        db.commit()
        logger.info(f"[DB] Saved quiz session for {video_stem}: {correct_answers}/{total_questions} ({accuracy:.1f}%)")
    except Exception as exc:
        db.rollback()
        logger.error(f"[DB] save_quiz_session FAILED: {exc}", exc_info=True)
    finally:
        db.close()


def get_study_recommendations(user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Generate "Today's Focus" study recommendations based on:
      1. Lowest quiz scores (needs improvement)
      2. Media with flashcards but no quiz attempts (untested)
      3. Oldest unreviewed flashcard sets (stale knowledge)
    Returns a list of recommendation dicts sorted by priority.
    """
    _, Media, Video, Image, Document, Flashcard, QuizQuestion, _, Progress, _, _, QuizSession = _models()
    db = _db()
    try:
        if isinstance(user_id, str):
            try:
                user_uuid = uuid.UUID(user_id)
            except ValueError:
                user_uuid = user_id
        else:
            user_uuid = user_id

        recommendations = []

        # Helper: get title for a media row
        def _title(media_row):
            t = media_row.batch_stem or "Unknown"
            if media_row.media_type in ("video", "live"):
                v = db.query(Video).filter(Video.media_id == media_row.id).first()
                if v and v.lecture_title:
                    t = v.lecture_title
            elif media_row.media_type == "image":
                img = db.query(Image).filter(Image.media_id == media_row.id).first()
                if img and img.lecture_title:
                    t = img.lecture_title
            elif media_row.media_type == "document":
                doc = db.query(Document).filter(Document.media_id == media_row.id).first()
                if doc and doc.lecture_title:
                    t = doc.lecture_title
            return t

        # ── 1. Lowest quiz scores ──────────────────────────────────────────
        from sqlalchemy import func as sqlfunc
        worst_quizzes = (
            db.query(
                QuizSession.media_id,
                sqlfunc.min(QuizSession.accuracy_score).label("worst_score"),
                sqlfunc.max(QuizSession.completed_at).label("last_attempt"),
            )
            .filter(QuizSession.user_id == user_uuid)
            .group_by(QuizSession.media_id)
            .having(sqlfunc.min(QuizSession.accuracy_score) < 70)
            .order_by(sqlfunc.min(QuizSession.accuracy_score).asc())
            .limit(limit)
            .all()
        )

        seen_media = set()
        for media_id, worst_score, last_attempt in worst_quizzes:
            m = db.query(Media).filter(Media.id == media_id).first()
            if not m:
                continue
            seen_media.add(media_id)
            urgency = "critical" if worst_score < 40 else "high" if worst_score < 60 else "medium"
            recommendations.append({
                "stem": m.batch_stem,
                "title": _title(m),
                "reason": f"Scored {worst_score:.0f}% — needs revision",
                "urgency": urgency,
                "type": "low_score",
                "score": worst_score,
                "icon": "alert",
            })

        # ── 2. Media with flashcards but never quizzed ─────────────────────
        all_media = db.query(Media).filter(Media.user_id == user_uuid).all()
        for m in all_media:
            if m.id in seen_media:
                continue
            fc_count = db.query(Flashcard).filter(Flashcard.media_id == m.id).count()
            qz_count = db.query(QuizQuestion).filter(QuizQuestion.media_id == m.id).count()
            session_count = db.query(QuizSession).filter(
                QuizSession.media_id == m.id, QuizSession.user_id == user_uuid
            ).count()

            if (fc_count > 0 or qz_count > 0) and session_count == 0:
                seen_media.add(m.id)
                recommendations.append({
                    "stem": m.batch_stem,
                    "title": _title(m),
                    "reason": f"{qz_count} quiz questions — never attempted",
                    "urgency": "medium",
                    "type": "untested",
                    "score": 0,
                    "icon": "book",
                })

        # ── 3. Oldest unreviewed flashcard sets ────────────────────────────
        for m in all_media:
            if m.id in seen_media:
                continue
            fc_count = db.query(Flashcard).filter(Flashcard.media_id == m.id).count()
            if fc_count == 0:
                continue
            review_count = db.query(Progress).filter(
                Progress.media_id == m.id,
                Progress.user_id == user_uuid,
                Progress.item_type == "flashcard",
            ).count()
            if review_count == 0:
                seen_media.add(m.id)
                recommendations.append({
                    "stem": m.batch_stem,
                    "title": _title(m),
                    "reason": f"{fc_count} flashcards — never reviewed",
                    "urgency": "low",
                    "type": "unreviewed",
                    "score": 0,
                    "icon": "cards",
                })

        # Sort: critical first, then high, medium, low
        urgency_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda r: (urgency_order.get(r["urgency"], 9), r["score"]))

        return recommendations[:limit]

    except Exception as exc:
        logger.error(f"[DB] get_study_recommendations FAILED: {exc}", exc_info=True)
        return []
    finally:
        db.close()


def get_lecture_context_for_plan(user_id: str, stem: str) -> Optional[Dict[str, Any]]:
    """
    Gather detailed context for a specific lecture to feed into the LLM for a study plan.
    """
    _, Media, Video, Image, Document, Flashcard, QuizQuestion, Note, Progress, _, _, QuizSession = _models()
    db = _db()
    try:
        if isinstance(user_id, str):
            try: user_uuid = uuid.UUID(user_id)
            except: user_uuid = user_id
        else: user_uuid = user_id

        # Find media
        media = db.query(Media).filter(Media.batch_stem == stem, Media.user_id == user_uuid).order_by(Media.uploaded_at.desc()).first()
        if not media:
            return None

        # Basic Info
        ctx = {
            "title": stem,
            "subject": "Unknown",
            "type": media.media_type,
            "summary": "",
            "topics": [],
            "concepts": [],
            "quiz_scores": [],
            "flashcard_stats": {"total": 0, "reviewed": 0, "avg_confidence": 0}
        }

        # Detailed Info from extension tables
        if media.media_type in ("video", "live"):
            v = db.query(Video).filter(Video.media_id == media.id).first()
            if v:
                ctx["title"] = v.lecture_title or ctx["title"]
                ctx["subject"] = v.subject_area or ctx["subject"]
                ctx["summary"] = v.summary or ""
                ctx["topics"] = v.main_topics or []
                # Extract concepts from audio_analysis JSON
                ta = v.transcription or {}
                if isinstance(ta, dict):
                    kc = ta.get("key_concepts", [])
                    if kc and isinstance(kc, list):
                        if kc and isinstance(kc[0], dict):
                            ctx["concepts"] = [item.get("concept") for item in kc if item.get("concept")]
                        else:
                            ctx["concepts"] = kc
        elif media.media_type == "image":
            img = db.query(Image).filter(Image.media_id == media.id).first()
            if img:
                ctx["title"] = img.lecture_title or ctx["title"]
                ctx["subject"] = img.subject_area or ctx["subject"]
                ctx["summary"] = img.description or ""
                ctx["concepts"] = img.key_concepts or []
        elif media.media_type == "document":
            d = db.query(Document).filter(Document.media_id == media.id).first()
            if d:
                ctx["title"] = d.lecture_title or ctx["title"]
                ctx["subject"] = d.subject_area or ctx["subject"]
                ctx["summary"] = d.summary or ""
                ctx["topics"] = d.main_topics or []
                # Documents might not have concepts yet in this version, but topics work
                ctx["concepts"] = d.main_topics or []

        # Quiz History
        sessions = db.query(QuizSession).filter(QuizSession.media_id == media.id, QuizSession.user_id == user_uuid).order_by(QuizSession.completed_at.desc()).limit(5).all()
        ctx["quiz_scores"] = [s.accuracy_score for s in sessions]

        # Flashcard Progress
        ctx["flashcard_stats"]["total"] = db.query(Flashcard).filter(Flashcard.media_id == media.id).count()
        progress_rows = db.query(Progress).filter(Progress.media_id == media.id, Progress.user_id == user_uuid, Progress.item_type == "flashcard").all()
        ctx["flashcard_stats"]["reviewed"] = len(progress_rows)
        if progress_rows:
            ctx["flashcard_stats"]["avg_confidence"] = sum(p.confidence or 0 for p in progress_rows) / len(progress_rows)

        return ctx
    except Exception as e:
        logger.error(f"[DB] get_lecture_context_for_plan failed: {e}")
        return None
    finally:
        db.close()