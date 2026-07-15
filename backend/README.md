

from __future__ import annotations import asyncio import json import os import shutil import subprocess from pathlib import Path from typing import Any, Dict, List, Optional, Tuple import cv2 import numpy as np from fastapi import FastAPI, File, HTTPException, UploadFile from fastap

pasted

THIS WAS MY CODE FOR DIFFERENT PROJECT BUT NOW I WANT TO MAKE A DIFFRENT PROJECT BUT THE BASE IS SAME ..ITS ABOUT Multi-Modal Academic Intelligence System  WHICH IS EXTRACTS FRAMES FROM VIDEOS .... ANALYSE IT TP EXTRACT USEFUL INFORAMTION FOR STUDENTS AND REASONING USING LLM ....EXTRACTS AUDIO FROM THE VIDEO TRANSCRIBE IT AND PROCESS IT TO SUMMARIZE THE LEARNING AND DOCUMENT SUPPORT AND IMAGE SUPPORT AS WELL ......REFORM MY CODE TO PERFORM THESE THINGS

Mar 11

Check available skills
Check available skills
Let me clarify the scope before writing this large codebase:

It seems the interactive widget isn't loading for you. Let me just ask directly — I'll make reasonable defaults and you can tell me to adjust:

I'll build the system with these assumptions:

LLM: Keep LlamaReasoner (local, easily swappable)
Detection: YOLO + OCR (same as your base)
Outputs: JSON API + Markdown study notes + Flashcards/Q&A
Let me now write the full reformed codebase:


Created 5 files, read a file
Created 5 files, read a file
Here's what was built — 4 files, drop-in replacement for your existing codebase:

main.py — The fully reformed FastAPI app. All surveillance-specific logic (telemetry, target tracking, mission intent) has been replaced with academic extraction. Key differences:

Per-frame prompt now extracts slide_title, key_concepts, definitions, formulas, bullet_points, diagram_type instead of telemetry
Audio pipeline now runs Whisper → topic/concept extraction via LLM (not just YAMNet event detection)
Two new generation steps at end of pipeline: Markdown study notes + Q&A flashcards
Two new endpoints: GET /results/notes/{stem} and GET /results/flashcards/{stem}
academic_system/yamnet_runner.py — YAMNet logic pulled out cleanly into its own module (same algorithm as yours, just isolated).

academic_system/llm_extensions.py — Adds a reason_text() method to LlamaReasoner so it can return raw Markdown (not just parsed JSON) for study notes generation.

README.md — Full architecture diagram, endpoint table, example outputs, and a diff table showing exactly what changed from the surveillance version.

Your existing video_pipeline/, metadata/, and models/yamnet/ directories stay completely untouched.

Main
PY 
Yamnet runner
PY 
Llm extensions
PY 
Readme
Document · MD 
Q: Which LLM should be used for reasoning/summarization?
A: Keep LlamaReasoner (local Llama)

Q: Which OCR/detection approach for academic content extraction?
A: Only OCR, no object detection

Q: What output formats should be generated for students? (Select all that apply)
A: JSON API results, Markdown study notes, PDF summary report, Structured flashcards/Q&A

Mar 11

Check PDF skill
Check PDF skill
Good, now I have exactly what I need. Let me rebuild everything precisely per your specs: LlamaReasoner only, OCR-only (no YOLO), all 4 output formats including PDF.


Ran 2 commands, created 2 files
Ran 2 commands, created 2 files
Now update the README to reflect the exact spec:


Ran a command, created a file, read a file
Ran a command, created a file, read a file
Here's exactly what was built per your spec:

LlamaReasoner (local Llama) — all reasoning stays on-device. No OpenAI/Anthropic API calls anywhere. Added llm_reason_text() to extract raw Markdown from the model (the base reason() method only returns parsed JSON).

OCR only — YOLO removed — YOLODetector, MultiStreamTracker, and telemetry_parser are completely gone. The pipeline now runs: frame → OCRExtractor → plain text → LlamaReasoner → academic JSON.

All 4 output formats:

#	Format	How
① JSON	Per-frame academic_content + lecture summary	GET /results/video
② Markdown	Full study notes with concepts, defs, formulas, summary	GET /results/notes/{stem}
③ PDF	Multi-section reportlab document (cover → concepts → defs table → formulas → slides → transcript → flashcards → reading list)	GET /results/pdf/{stem} ← file download
④ Flashcards	15–20 Q&A JSON cards	GET /results/flashcards/{stem}
The only dependency addition is pip install reportlab for the PDF. Everything else was already in your stack.

Main
PY 
Yamnet runner
PY 
Readme
Document · MD 
project/
├── main.py                          ← This file (FastAPI app)
├── academic_system/
│   ├── __init__.py
│   ├── llm_extensions.py            ← reason_text() patch for LlamaReasoner
│   └── yamnet_runner.py             ← Standalone YAMNet inference
├── video_pipeline/                  ← Existing pipeline (unchanged)
│   ├── config.py
│   ├── core/stream_manager.py
│   ├── detection/detector.py
│   ├── detection/ocr.py
│   ├── detection/tracker.py
│   └── reasoning/llama_engine.py
├── metadata/video_metadata.py       ← Existing metadata extractor
├── models/
│   └── yamnet/
│       ├── yamnet.h5
│       ├── yamnet.py
│       ├── params.py
│       ├── features.py
│       └── yamnet_class_map.csv
├── uploads/                         ← Uploaded videos
├── audio/                           ← Extracted audio (WAV)
├── images/                          ← Uploaded images
├── extracted_frames/                ← Saved JPEG frames
│   └── {video_stem}/
│       ├── frame_00001_t0.000s.jpg
│       └── frames_index.json
├── study_notes/                     ← Generated Markdown notes
│   └── {video_stem}_study_notes.md
└── flashcards/                      ← Generated flashcard JSON
    └── {video_stem}_flashcards.json     GENERATE CODE FOR EACH AND EVERY FILE NO NEED OF YAMNET AS LECTURE VIDEOS DONT NEED THESE ....ALSO ADD YOUR SUGGESTIONS WHAT ELSE CAN BE ADDED TO IMPROVE THIS PROJECT ...

Show less
Mar 11
Let me read all the relevant files first to understand the full codebase before generating everything.


Viewed 2 files
Viewed 2 files
Good. Now I have the full picture. Let me generate every single file cleanly — no YAMNet, complete code for every module.


Ran 2 commands
Ran 2 commands
Now generating every file:


Created 20 files, ran a command
Created 20 files, ran a command
All 20 files generated. Let me present them:


Presented 13 files
Presented 13 files
Here's every file generated — 20 total, zero stubs, zero placeholders:

File	What it does
main.py	FastAPI app — full pipeline orchestration, all endpoints
academic_system/prompts.py	Every LLM prompt in one file — easy to tune without touching logic
academic_system/pdf_generator.py	Full ReportLab PDF (8 sections: cover → concepts → formulas → slides → transcript → flashcards → reading list)
academic_system/whisper_transcriber.py	Audio extraction (ffmpeg) + Whisper transcription, model cached per process
video_pipeline/config.py	All config as a dataclass — every value overridable via env var
video_pipeline/core/stream_manager.py	Multi-stream video reader, seek-based FPS sampling, async-safe
video_pipeline/detection/ocr.py	EasyOCR primary + Tesseract fallback, with grayscale pre-processing
video_pipeline/reasoning/llama_engine.py	LlamaReasoner with reason() (JSON) and reason_text() (Markdown), 4-bit quant support
video_pipeline/utils/device.py	CUDA → MPS → CPU auto-detection
video_pipeline/utils/logger.py	Centralised logging factory
metadata/video_metadata.py	cv2 + ffprobe metadata extraction
requirements.txt	All Python deps with versions, system deps documented
README.md	Full docs + 14 concrete improvement suggestions
Key improvements added beyond your spec: a bonus ⑤ MCQ Quiz output (GET /results/quiz/{stem}) and a MIN_OCR_CHARS guard that skips the LLM entirely on blank/transition frames — saves significant processing time on real lecture videos.

Main
PY 
Prompts
PY 
Pdf generator
PY 
Whisper transcriber
PY 
Config
PY 
Stream manager
PY 
Ocr
PY 
Llama engine
PY 
Device
PY 
Logger
PY 
Video metadata
PY 
Requirements
TXT 
Readme
Document · MD 





Claude is AI and can make mistakes. Please double-check responses.
Readme · MD
Copy

Multi-Modal Academic Intelligence System v2.0.0
Transforms lecture videos, slide images, and audio recordings into structured student learning materials — all running locally, no cloud APIs.

Stack
Component	Technology	Notes
Reasoning / LLM	LlamaReasoner (local Llama)	HuggingFace Transformers, optional 4-bit quant
OCR	EasyOCR (primary) + Tesseract (fallback)	GPU-capable, no YOLO
Speech	openai-whisper	Local model, no API key
PDF	ReportLab Platypus	Multi-section academic report
Frame extraction	OpenCV StreamManager	Configurable target FPS
Web API	FastAPI + uvicorn	Async pipeline, CORS enabled
YAMNet and YOLO have been fully removed — not needed for academic lecture content.

Output Formats — per video
#	Format	Endpoint	File saved
①	JSON academic analysis	GET /results/video	in-memory
②	Markdown study notes	GET /results/notes/{stem}	study_notes/{stem}_study_notes.md
③	PDF report (download)	GET /results/pdf/{stem}	pdf_reports/{stem}_academic_report.pdf
④	Q&A flashcards	GET /results/flashcards/{stem}	flashcards/{stem}_flashcards.json
⑤	MCQ quiz	GET /results/quiz/{stem}	quizzes/{stem}_quiz.json
Project Structure
project/
│
├── main.py                                  ← FastAPI app — pipeline orchestration
│
├── academic_system/                         ← Academic-specific logic
│   ├── __init__.py
│   ├── prompts.py                           ← All LLM prompt builders (one place)
│   ├── pdf_generator.py                     ← ReportLab PDF generation
│   └── whisper_transcriber.py               ← Audio extraction + Whisper transcription
│
├── video_pipeline/                          ← Core pipeline infrastructure
│   ├── __init__.py
│   ├── config.py                            ← All config (env-overridable)
│   ├── core/
│   │   ├── __init__.py
│   │   └── stream_manager.py                ← Multi-stream video frame reader
│   ├── detection/
│   │   ├── __init__.py
│   │   └── ocr.py                           ← EasyOCR + Tesseract extractor
│   ├── reasoning/
│   │   ├── __init__.py
│   │   └── llama_engine.py                  ← LlamaReasoner: reason() + reason_text()
│   └── utils/
│       ├── __init__.py
│       ├── device.py                        ← CUDA / MPS / CPU detection
│       └── logger.py                        ← Centralised logging
│
├── metadata/
│   ├── __init__.py
│   └── video_metadata.py                    ← Video technical metadata (cv2 + ffprobe)
│
├── requirements.txt
├── README.md
│
├── uploads/                                 ← Uploaded video files
├── audio/                                   ← Extracted WAV audio files
├── images/                                  ← Uploaded image files
├── extracted_frames/
│   └── {video_stem}/
│       ├── frame_00001_t0.000s.jpg
│       └── frames_index.json
├── study_notes/
│   └── {video_stem}_study_notes.md          ← ② Markdown study notes
├── pdf_reports/
│   └── {video_stem}_academic_report.pdf     ← ③ PDF report
├── flashcards/
│   └── {video_stem}_flashcards.json         ← ④ Q&A flashcards
├── quizzes/
│   └── {video_stem}_quiz.json               ← ⑤ MCQ quiz
└── outputs/                                 ← General pipeline outputs
Pipeline Flow
POST /upload/video
│
├─ PHASE 1: Frame-level (per sampled frame at target FPS)
│   ├─ StreamManager → BGR frame + timestamp
│   ├─ OCRExtractor  → raw OCR results → plain text
│   ├─ [skip if text < MIN_OCR_CHARS — blank/transition slide]
│   ├─ LlamaReasoner.reason(prompt_frame_extract)
│   │     → { slide_title, key_concepts, definitions,
│   │         formulas, bullet_points, importance }
│   ├─ Save JPEG → extracted_frames/{stem}/
│   └─ Update frames_index.json (incremental)
│
└─ PHASE 2: Per-video (after all frames)
    ├─ 2a. extract_audio() → ffmpeg → 16kHz mono WAV
    │       whisper_transcriber.transcribe() → full transcript + segments
    ├─ 2b. LlamaReasoner.reason(prompt_audio_topics)
    │       → { topics_covered, key_concepts, summary, suggested_reading }
    ├─ 2c. LlamaReasoner.reason(prompt_lecture_summary)
    │       → { main_topics, learning_outcomes, summary, difficulty_level }
    ├─ 2d. LlamaReasoner.reason_text(prompt_study_notes)
    │       → Markdown study notes → study_notes/{stem}_study_notes.md
    ├─ 2e. LlamaReasoner.reason(prompt_flashcards)
    │       → 15-20 Q&A cards → flashcards/{stem}_flashcards.json
    ├─ 2f. LlamaReasoner.reason(prompt_quiz)
    │       → 10 MCQ questions → quizzes/{stem}_quiz.json
    └─ 2g. generate_pdf_report()
            → pdf_reports/{stem}_academic_report.pdf
API Endpoints
Upload
Method	Endpoint	Description
POST	/upload/video	1–3 lecture videos → full pipeline
POST	/upload/image	1–3 slide images → immediate OCR + LLM
POST	/upload/audio	1–3 audio files → Whisper transcription
Results
Method	Endpoint	Description
GET	/status	Pipeline progress + per-video readiness flags
GET	/results/video	Full JSON for all videos
GET	/results/image	Image analysis results
GET	/results/audio	Transcription results
GET	/results/notes/{stem}	Markdown study notes
GET	/results/pdf/{stem}	PDF download
GET	/results/flashcards/{stem}	Q&A flashcards JSON
GET	/results/quiz/{stem}	MCQ quiz JSON
GET	/results/frames/{stem}	Frame index
GET	/results/latest?n=10	Last N frames
Control
Method	Endpoint	Description
POST	/stop	Stop pipeline (outputs generated from collected frames)
DELETE	/results	Clear all in-memory results
GET	/diagnostics	Health check + endpoint map
GET	/docs	Swagger UI
Setup
bash
# 1. System dependencies
sudo apt install ffmpeg tesseract-ocr   # Ubuntu/Debian
# brew install ffmpeg tesseract          # macOS

# 2. Python dependencies
pip install -r requirements.txt

# 3. Run
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
Environment Variables
Variable	Default	Description
REASONING_MODEL_ID	meta-llama/Meta-Llama-3-8B-Instruct	HuggingFace model ID or local path
MAX_REASONING_TOKENS	1024	Max LLM output tokens per call
WHISPER_MODEL_SIZE	base	tiny / base / small / medium / large
PIPELINE_FPS	1.0	Frames per second to sample from video
MIN_OCR_CHARS	20	Min OCR characters to trigger LLM call
OCR_CONFIDENCE_THRESHOLD	0.4	Discard OCR results below this confidence
SERVER_HOST	127.0.0.1	Host for frame URL generation
SERVER_PORT	8000	Port for frame URL generation
LOG_LEVEL	INFO	DEBUG / INFO / WARNING / ERROR
Data Schemas
Per-frame academic_content
json
{
  "slide_title":     "Newton's Laws of Motion",
  "key_concepts":    ["inertia", "force", "mass", "acceleration"],
  "definitions":     [{"term": "force", "definition": "..."}],
  "formulas":        ["F = ma", "a = F/m"],
  "bullet_points":   ["An object at rest stays at rest unless acted on by a force"],
  "diagram_type":    "equation",
  "content_summary": "Introduction to Newton's second law.",
  "importance":      "high"
}
Flashcard
json
{
  "question":   "What does Newton's second law state?",
  "answer":     "Force equals mass times acceleration: F = ma",
  "topic":      "Classical Mechanics",
  "difficulty": "easy"
}
MCQ Quiz question
json
{
  "question":       "What is the SI unit of force?",
  "options":        {"A": "Joule", "B": "Newton", "C": "Watt", "D": "Pascal"},
  "correct_answer": "B",
  "explanation":    "The Newton (N) is the SI unit of force, defined as kg·m/s².",
  "topic":          "Units and Measurements"
}
PDF Report Sections
Cover page — title, subject, difficulty, file info, counts
Lecture Overview — summary paragraph, topics, learning outcomes
Key Concepts & Definitions — concept explanations + definitions table
Formulas & Equations — all unique formulas across frames
Slide-by-Slide Content — high/medium importance frames
Audio Transcript Excerpt — formatted Whisper output
Flashcards — full Q&A section printed in document
Suggested Reading — books and topics from audio analysis
Suggested Improvements for Future Versions
🎯 High Impact
Slide change detection — use frame differencing (histogram or SSIM) to detect actual slide transitions instead of sampling at fixed FPS. This avoids duplicate OCR on identical frames and ensures every unique slide is captured exactly once.
Semantic deduplication — embed key concepts using a sentence transformer (e.g. all-MiniLM-L6-v2) and cluster similar ones. Prevents the same idea appearing 10 times in study notes from repeated slide content.
Speaker diarization — use pyannote-audio to separate the lecturer's voice from student questions, enabling per-speaker transcript segments. Improves the quality of topic extraction.
Knowledge graph — build a concept-relationship graph per lecture using NetworkX. Expose it as a JSON endpoint (GET /results/graph/{stem}) and render it as an interactive D3.js visualization in a companion frontend.
📊 Content Quality
Math OCR — use pix2tex or LaTeX-OCR for equation-dense slides (physics, maths). Standard OCR struggles with mathematical notation; a dedicated model produces proper LaTeX output.
Diagram understanding — run a lightweight vision model (e.g. BLIP-2 or LLaVA) on frames classified as diagram_type != none to generate natural language descriptions of charts, flowcharts, and graphs.
Timestamp-linked study notes — embed [timestamp: 4m32s] links in the Markdown notes next to each concept, allowing students to jump to the exact video moment. Requires a video player frontend.
Spaced repetition scheduling — add an Anki-compatible export format (.apkg) for the flashcards so students can import them directly into Anki for optimally timed review.
🔧 System & Scale
Celery + Redis task queue — move run_academic_pipeline from an asyncio task to a Celery worker. This allows multiple videos to be processed in parallel across multiple machines and survives server restarts.
Database persistence — store results in SQLite (via SQLAlchemy) instead of an in-memory dict. Results survive restarts, can be queried by student, and enable a course-level aggregation view.
Multi-language support — pass the detected language field from Whisper back into the OCR extractor and LLM prompts. EasyOCR already supports 80+ languages; the prompts need only minor localisation.
Student progress API — add endpoints to track which flashcards a student has reviewed and their self-assessed confidence (PATCH /flashcards/{stem}/{card_id}/review), enabling adaptive learning loops.
🖥️ Frontend
React dashboard — a companion Next.js/React app consuming these endpoints, showing: video upload form, live status bar, rendered Markdown notes, interactive flashcard reviewer, quiz mode, and PDF download button.
Video player with annotations — embed a video player (e.g. Video.js) that overlays the detected slide titles and key concepts at the correct timestamps as the student watches.
