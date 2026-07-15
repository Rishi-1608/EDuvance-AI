"""
AcademIQ — Next-Gen Database Schema v4.0.0
==========================================
Universal intake architecture with one-to-one and one-to-many extensions.
All primary keys are UUIDs.
"""

from __future__ import annotations
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer,
    String, Text, UniqueConstraint, Index, func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
)
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
#  Connection
# ──────────────────────────────────────────────────────────────────────────────

# Managed Postgres providers (Neon, Supabase, etc.) hand you one full
# connection string instead of separate host/user/pass fields, and Neon in
# particular requires SSL. If DATABASE_URL is set, it's used as-is (this is
# the simplest path for Neon — just paste its connection string). Otherwise
# we fall back to building one from the individual DB_* vars, for local dev
# against a plain local Postgres with no SSL requirement.
DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    POSTGRES_URL = DATABASE_URL
else:
    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_PORT = os.environ.get("DB_PORT", "5432")
    DB_NAME = os.environ.get("DB_NAME", "eduvance_db")
    DB_USER = os.environ.get("DB_USER", "postgres")
    DB_PASS = os.environ.get("DB_PASS", "postgres")
    DB_SSLMODE = os.environ.get("DB_SSLMODE")  # e.g. "require" for Neon

    POSTGRES_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    if DB_SSLMODE:
        POSTGRES_URL += f"?sslmode={DB_SSLMODE}"

engine = create_engine(
    POSTGRES_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=1800,
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

# ──────────────────────────────────────────────────────────────────────────────
#  Models
# ──────────────────────────────────────────────────────────────────────────────

class User(Base):
    """Stores every registered account. Root of all data."""
    __tablename__ = "users"

    id:            Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username:      Mapped[str]       = mapped_column(String(64), unique=True, index=True, nullable=False)
    email:         Mapped[str]       = mapped_column(String(256), unique=True, index=True, nullable=False)
    password_hash: Mapped[str]       = mapped_column(String(256), nullable=False)
    created_at:    Mapped[datetime]  = mapped_column(DateTime, server_default=func.now())
    updated_at:    Mapped[datetime]  = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    media:    Mapped[List["Media"]]    = relationship(back_populates="owner", cascade="all, delete-orphan")
    progress: Mapped[List["Progress"]] = relationship(back_populates="user",  cascade="all, delete-orphan")

class Media(Base):
    """The universal intake table. Record of every upload."""
    __tablename__ = "media"

    id:                Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id:           Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    media_type:        Mapped[str]       = mapped_column(String(32), nullable=False) # "video", "image", "document"
    original_filename: Mapped[str]       = mapped_column(String(512), nullable=False)
    storage_path:      Mapped[str]       = mapped_column(String(1024), nullable=False)
    minio_object_key:  Mapped[str]       = mapped_column(String(1024), nullable=False)
    file_size_bytes:   Mapped[int]       = mapped_column(Integer, nullable=False)
    batch_stem:        Mapped[Optional[str]] = mapped_column(String(256), nullable=True) # Groups image batches
    uploaded_at:       Mapped[datetime]  = mapped_column(DateTime, server_default=func.now())

    # Relationships
    owner:     Mapped["User"]     = relationship(back_populates="media")
    video:     Mapped[Optional["Video"]]    = relationship(back_populates="media", uselist=False, cascade="all, delete-orphan")
    document:  Mapped[Optional["Document"]] = relationship(back_populates="media", uselist=False, cascade="all, delete-orphan")
    images:    Mapped[List["Image"]]        = relationship(back_populates="media", cascade="all, delete-orphan")
    
    flashcards:     Mapped[List["Flashcard"]]    = relationship(back_populates="media", cascade="all, delete-orphan")
    quiz_questions: Mapped[List["QuizQuestion"]] = relationship(back_populates="media", cascade="all, delete-orphan")
    notes:          Mapped[List["Note"]]         = relationship(back_populates="media", cascade="all, delete-orphan")
    progress:       Mapped[List["Progress"]]     = relationship(back_populates="media", cascade="all, delete-orphan")
    transcription_segments: Mapped[List["TranscriptionSegment"]] = relationship(back_populates="media", cascade="all, delete-orphan")
    result_stats:   Mapped[Optional["MediaResultStats"]] = relationship(back_populates="media", uselist=False, cascade="all, delete-orphan")

class Video(Base):
    """Extends media for video uploads (one-to-one)."""
    __tablename__ = "videos"

    id:               Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    media_id:         Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("media.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    duration_sec:     Mapped[float]     = mapped_column(Float, nullable=True)
    adaptive_fps:     Mapped[float]     = mapped_column(Float, nullable=True)
    whisper_model:    Mapped[str]       = mapped_column(String(64), nullable=True)
    detected_language: Mapped[str]       = mapped_column(String(32), nullable=True)
    
    # Phase 2 LLM outputs
    lecture_title:     Mapped[str]       = mapped_column(String(512), nullable=True)
    subject_area:      Mapped[str]       = mapped_column(String(256), nullable=True)
    difficulty_level:  Mapped[str]       = mapped_column(String(64), nullable=True)
    summary:           Mapped[str]       = mapped_column(Text, nullable=True)
    main_topics:       Mapped[dict]       = mapped_column(JSONB, nullable=True) # List of topics
    learning_outcomes: Mapped[dict]       = mapped_column(JSONB, nullable=True) # List of outcomes
    transcription:     Mapped[dict]       = mapped_column(JSONB, nullable=True) # Full transcription object
    
    study_notes_path:     Mapped[str]    = mapped_column(String(1024), nullable=True)
    pdf_report_path:      Mapped[str]    = mapped_column(String(1024), nullable=True)
    knowledge_graph_path: Mapped[str]    = mapped_column(String(1024), nullable=True)
    processed_at:         Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    media: Mapped["Media"] = relationship(back_populates="video")

class Image(Base):
    """Extends media for image uploads (one-to-many if batch)."""
    __tablename__ = "images"

    id:               Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    media_id:         Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("media.id", ondelete="CASCADE"), nullable=False)
    
    batch_stem:       Mapped[str]       = mapped_column(String(256), nullable=False)
    batch_index:      Mapped[int]       = mapped_column(Integer, nullable=False)
    
    # process_image_academic outputs
    ocr_text:         Mapped[str]       = mapped_column(Text, nullable=True)
    image_title:      Mapped[str]       = mapped_column(String(512), nullable=True)
    content_type:     Mapped[str]       = mapped_column(String(128), nullable=True)
    description:      Mapped[str]       = mapped_column(Text, nullable=True)
    key_concepts:     Mapped[dict]       = mapped_column(JSONB, nullable=True)
    bullet_points:    Mapped[dict]       = mapped_column(JSONB, nullable=True)
    formulas:         Mapped[dict]       = mapped_column(JSONB, nullable=True)
    
    # Batch-level metadata (duplicated)
    lecture_title:    Mapped[str]       = mapped_column(String(512), nullable=True)
    subject_area:     Mapped[str]       = mapped_column(String(256), nullable=True)
    study_notes_path: Mapped[str]       = mapped_column(String(1024), nullable=True)
    pdf_report_path:  Mapped[str]       = mapped_column(String(1024), nullable=True)
    processed_at:     Mapped[datetime]  = mapped_column(DateTime, server_default=func.now())

    media: Mapped["Media"] = relationship(back_populates="images")

class Document(Base):
    """Extends media for PDF uploads (one-to-one)."""
    __tablename__ = "documents"

    id:               Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    media_id:         Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("media.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    page_count:       Mapped[int]       = mapped_column(Integer, nullable=True)
    
    # Mirrors videos structure
    lecture_title:     Mapped[str]       = mapped_column(String(512), nullable=True)
    subject_area:      Mapped[str]       = mapped_column(String(256), nullable=True)
    difficulty_level:  Mapped[str]       = mapped_column(String(64), nullable=True)
    summary:           Mapped[str]       = mapped_column(Text, nullable=True)
    main_topics:       Mapped[dict]       = mapped_column(JSONB, nullable=True)
    learning_outcomes: Mapped[dict]       = mapped_column(JSONB, nullable=True)
    
    study_notes_path:     Mapped[str]    = mapped_column(String(1024), nullable=True)
    pdf_report_path:      Mapped[str]    = mapped_column(String(1024), nullable=True)
    knowledge_graph_path: Mapped[str]    = mapped_column(String(1024), nullable=True)
    processed_at:         Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    media: Mapped["Media"] = relationship(back_populates="document")

class Flashcard(Base):
    """Stores every generated Q&A card."""
    __tablename__ = "flashcards"

    id:           Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    media_id:     Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("media.id", ondelete="CASCADE"), nullable=False)
    
    card_index:   Mapped[int]       = mapped_column(Integer, nullable=False)
    question:     Mapped[str]       = mapped_column(Text, nullable=False)
    answer:       Mapped[str]       = mapped_column(Text, nullable=False)
    topic:        Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    difficulty:   Mapped[str]       = mapped_column(String(16), nullable=False, default="medium")
    created_at:   Mapped[datetime]  = mapped_column(DateTime, server_default=func.now())

    media: Mapped["Media"] = relationship(back_populates="flashcards")

    __table_args__ = (
        UniqueConstraint("media_id", "card_index", name="uq_flashcard_index"),
    )

class QuizQuestion(Base):
    """Stores every generated MCQ question."""
    __tablename__ = "quiz_questions"

    id:             Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    media_id:       Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("media.id", ondelete="CASCADE"), nullable=False)
    
    question_index: Mapped[int]       = mapped_column(Integer, nullable=False)
    question:       Mapped[str]       = mapped_column(Text, nullable=False)
    options:        Mapped[dict]      = mapped_column(JSONB, nullable=False) # {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct_answer: Mapped[str]       = mapped_column(String(8), nullable=False)
    explanation:    Mapped[str]       = mapped_column(Text, nullable=True)
    topic:          Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    created_at:     Mapped[datetime]  = mapped_column(DateTime, server_default=func.now())

    media: Mapped["Media"] = relationship(back_populates="quiz_questions")

    __table_args__ = (
        UniqueConstraint("media_id", "question_index", name="uq_quiz_index"),
    )

class Note(Base):
    """Stores every generated study note."""
    __tablename__ = "notes"

    id:           Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    media_id:     Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("media.id", ondelete="CASCADE"), nullable=False)
    
    content:      Mapped[str]       = mapped_column(Text, nullable=False)
    created_at:   Mapped[datetime]  = mapped_column(DateTime, server_default=func.now())
    updated_at:   Mapped[datetime]  = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    media: Mapped["Media"] = relationship(back_populates="notes")

class TranscriptionSegment(Base):
    """Stores granular transcription segments for fine-grained retrieval."""
    __tablename__ = "transcription_segments"

    id:            Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    media_id:      Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("media.id", ondelete="CASCADE"), nullable=False, index=True)
    
    segment_index: Mapped[int]       = mapped_column(Integer, nullable=False)
    start_time:    Mapped[float]     = mapped_column(Float, nullable=False)
    end_time:      Mapped[float]     = mapped_column(Float, nullable=False)
    text:          Mapped[str]       = mapped_column(Text, nullable=False)
    confidence:    Mapped[float]     = mapped_column(Float, nullable=True)

    media: Mapped["Media"] = relationship(back_populates="transcription_segments")

    __table_args__ = (
        UniqueConstraint("media_id", "segment_index", name="uq_segment_index"),
    )

class MediaResultStats(Base):
    """Stores summary stats for the UI 'Info Cards'."""
    __tablename__ = "media_result_stats"

    id:             Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    media_id:       Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("media.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    duration_sec:   Mapped[float]     = mapped_column(Float, nullable=True)
    segment_count:  Mapped[int]       = mapped_column(Integer, nullable=True)
    word_count:     Mapped[int]       = mapped_column(Integer, nullable=True)
    language_code:  Mapped[str]       = mapped_column(String(16), nullable=True)
    
    processed_at:   Mapped[datetime]  = mapped_column(DateTime, server_default=func.now())

    media: Mapped["Media"] = relationship(back_populates="result_stats")

class Progress(Base):
    """Records review attempts by a user."""
    __tablename__ = "progress"

    id:           Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id:      Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    media_id:     Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("media.id", ondelete="CASCADE"), nullable=False, index=True)
    
    item_type:    Mapped[str]       = mapped_column(String(32), nullable=False) # "flashcard" or "quiz_question"
    item_id:      Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False) # UUID of specific card or question
    
    confidence:   Mapped[Optional[int]] = mapped_column(Integer, nullable=True) # 1-5 for flashcards
    correct:      Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True) # for quiz questions
    session_id:   Mapped[str]       = mapped_column(String(128), nullable=True)
    reviewed_at:  Mapped[datetime]  = mapped_column(DateTime, server_default=func.now())

    user:  Mapped["User"]  = relationship(back_populates="progress")
    media: Mapped["Media"] = relationship(back_populates="progress")

# ──────────────────────────────────────────────────────────────────────────────
#  DB Init
# ──────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    # This will create tables if they don't exist. 
    # Since the user asked to "DELETE WHATEVER REQUIRED", we might want to drop first,
    # but that's destructive. Usually we should just create.
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
