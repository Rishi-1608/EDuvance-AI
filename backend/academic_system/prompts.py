"""
academic_system/prompts.py
============================
All LLM prompt-builder functions for the Academic Intelligence System.

v4 changes (2-call split fix)
------------------------------
- Added prompt_analysis_and_notes(): Call 1 of the new 2-call Phase 2 path.
  Produces lecture_title, subject_area, difficulty, topics, learning_outcomes,
  summary, and study_notes in one pass (~1,400 token output budget).
  Transcript capped at 2,000 chars, frame timeline sampled to 6 rows.
  Output schema is compact but complete — study_notes is inline Markdown.

- Added prompt_flashcards_and_quiz(): Call 2 of the new 2-call Phase 2 path.
  Input is only the distilled fields from Call 1 (title, subject, concepts,
  important_points) — no raw frames, no transcript.  This keeps the prompt
  under ~500 tokens, leaving ~700 tokens for 8 flashcards + 5 quiz MCQs.
  Easily fits Phi-3-mini's 4 096-token context window.

- WHY the original prompt_everything() stalled:
  prompt_everything() packed transcript (2,500 chars ≈ 625 tokens) + frame
  timeline + concepts + the full output schema into one call, leaving only
  ~800–1,000 tokens for generation.  On RTX 3050 with 4-bit Phi-3-mini that
  was enough for the JSON wrapper but not the body, so the model either hit
  max_new_tokens mid-object (truncated JSON → parse failure → empty result)
  or timed out.  Splitting at the natural boundary (analysis/notes vs
  flashcards/quiz) gives each call a comfortable budget.

v3 changes
----------
- prompt_study_notes() and prompt_flashcards() now accept optional
  deduped_concepts, deduped_formulas, deduped_defs parameters.
  When provided they are injected into the prompt instead of the raw
  (potentially repetitive) per-frame data, producing cleaner outputs.
- All functions remain fully backwards-compatible — the new parameters
  default to None so existing call sites work without changes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
#  PER-FRAME
# ──────────────────────────────────────────────────────────────────────────────

def prompt_frame_extract(ocr_text: str, timestamp: float, frame_id: int) -> str:
    """Extract academic content from the OCR text of a single lecture frame."""
    return f"""You are an academic content extraction assistant analysing a frame from a lecture video.

Frame ID  : {frame_id}
Timestamp : {timestamp:.2f} seconds

=== TEXT DETECTED IN FRAME (OCR) ===
{ocr_text or "(no readable text found in this frame)"}

Instructions:
- Extract every piece of academically useful information from the OCR text above.
- If the frame appears to be a blank/transition slide with little or no text, return importance "low" and empty arrays for other fields.
- Be precise: only include what you can actually read from the OCR text, do not invent content.

Return ONLY a single valid JSON object. No markdown fences, no explanation, no preamble:

{{
  "slide_title":     "The exact title of this slide, or empty string if none",
  "key_concepts":    ["list", "of", "main", "concepts"],
  "definitions":     [{{"term": "term name", "definition": "definition text"}}],
  "formulas":        ["F = ma", "E = mc^2"],
  "bullet_points":   ["each bullet point as a separate string"],
  "diagram_type":    "none | flowchart | graph | table | equation | image | code",
  "content_summary": "One clear sentence summarising this frame's academic content.",
  "importance":      "high | medium | low"
}}"""


# ──────────────────────────────────────────────────────────────────────────────
#  IMAGE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def prompt_image_extract(ocr_text: str) -> str:
    """Comprehensive analysis of a standalone educational image or slide."""
    return f"""You are an academic content extraction assistant analysing an educational image or slide.

=== TEXT DETECTED IN IMAGE (OCR) ===
{ocr_text or "(no readable text detected)"}

Instructions:
- Extract all academically meaningful content from the OCR text.
- Write a thorough explanation of the content that would help a student understand it.
- Suggest 2-3 practical study tips related to this material.

Return ONLY a single valid JSON object. No markdown fences, no explanation:

{{
  "slide_title":          "Title of the slide or image",
  "subject_area":         "e.g. Physics, Computer Science, History",
  "key_concepts":         ["concept1", "concept2"],
  "definitions":          [{{"term": "...", "definition": "..."}}],
  "formulas":             ["..."],
  "bullet_points":        ["each key point"],
  "diagram_type":         "none | flowchart | graph | table | equation | image | code",
  "detailed_explanation": "A thorough paragraph explaining the content for a student who is encountering it for the first time.",
  "study_tips":           ["Tip 1", "Tip 2", "Tip 3"]
}}"""


# ──────────────────────────────────────────────────────────────────────────────
#  COMBINED ANALYSIS  (replaces two separate calls with one)
# ──────────────────────────────────────────────────────────────────────────────

def prompt_combined_analysis(
    video_path:     str,
    frame_analyses: List[Dict[str, Any]],
    transcript:     str,
    sample_n:       int = 10,
) -> str:
    """
    Single LLM call that replaces prompt_audio_topics() + prompt_lecture_summary().

    Saves one entire Phi-3 forward pass (~40-60 seconds on RTX 3050).

    Returns a JSON object with two top-level keys:
      "audio_analysis"    — same fields as prompt_audio_topics output
      "lecture_summary"   — same fields as prompt_lecture_summary output

    Both callers in main.py read their respective key after this call.
    """
    excerpt        = transcript[:4000]
    truncated_note = "\n[... transcript truncated ...]" if len(transcript) > 4000 else ""

    step      = max(1, len(frame_analyses) // sample_n)
    snapshots = []
    for fr in frame_analyses[::step]:
        ac = fr.get("academic_content", {})
        snapshots.append(
            f"  [{fr['timestamp']:.1f}s] "
            f"slide='{ac.get('slide_title', '?')}' "
            f"concepts={ac.get('key_concepts', [])[:3]}"
        )

    return f"""You are an academic content extraction assistant analysing a lecture video.

Video file : {Path(video_path).name}

=== TRANSCRIPT ===
{excerpt}{truncated_note}

=== SLIDE TIMELINE (sampled) ===
{chr(10).join(snapshots) or "  (no frames available)"}

Return ONLY a single valid JSON object with exactly these two keys. No markdown, no preamble:

{{
  "audio_analysis": {{
    "lecture_title":     "Inferred title from transcript",
    "subject_area":      "Academic subject (e.g. Physics, History)",
    "topics_covered":    ["Topic 1", "Topic 2"],
    "key_concepts":      [{{"concept": "name", "explanation": "brief explanation"}}],
    "important_points":  ["Key point students must remember"],
    "summary":           "3-4 sentence summary of what was taught.",
    "suggested_reading": ["Book or topic for further study"]
  }},
  "lecture_summary": {{
    "lecture_title":     "Full descriptive title of this lecture",
    "subject_area":      "Academic subject area",
    "main_topics":       ["Topic 1", "Topic 2", "Topic 3"],
    "learning_outcomes": ["Students will be able to ...", "Students will understand ..."],
    "summary":           "4-5 sentence paragraph summarising the entire lecture.",
    "difficulty_level":  "beginner | intermediate | advanced"
  }}
}}"""


def prompt_combined_outputs(
    video_path:       str,
    frame_analyses:   List[Dict[str, Any]],
    audio_topics:     Dict[str, Any],
    lecture_summary:  Dict[str, Any],
    deduped_concepts: Optional[List[str]]  = None,
    deduped_formulas: Optional[List[str]]  = None,
    deduped_defs:     Optional[List[Dict]] = None,
    max_concepts: int = 12,
    max_defs:     int = 6,
    max_formulas: int = 6,
) -> str:
    """
    Single LLM call that replaces prompt_flashcards() + prompt_quiz().

    Saves one entire Phi-3 forward pass (~40-60 seconds on RTX 3050).

    Returns a JSON object with two top-level keys:
      "flashcards"  — list of Q&A cards (same schema as before)
      "quiz"        — list of MCQ questions (same schema as before)

    Caps are tighter than the individual prompts to fit within context window.
    """
    concepts = (deduped_concepts or [])[:max_concepts]
    defs     = (deduped_defs     or [])[:max_defs]
    formulas = (deduped_formulas or [])[:max_formulas]

    audio_concepts = audio_topics.get("key_concepts", [])[:6]
    important_pts  = audio_topics.get("important_points", [])[:4]

    return f"""You are an expert academic content creator for a lecture on:
"{lecture_summary.get('lecture_title', Path(video_path).name)}"
Subject: {lecture_summary.get('subject_area', 'General')}

=== KEY CONCEPTS ===
{json.dumps(concepts, indent=2)}

=== FORMULAS ===
{json.dumps(formulas, indent=2)}

=== DEFINITIONS ===
{json.dumps(defs, indent=2)}

=== IMPORTANT POINTS ===
{json.dumps(important_pts, indent=2)}

=== AUDIO CONCEPTS ===
{json.dumps(audio_concepts, indent=2)}

Create BOTH flashcards AND a quiz in one response.

Return ONLY a single valid JSON object with exactly these two keys. No markdown, no preamble:

{{
  "flashcards": [
    {{
      "question":   "Specific question testing one concept",
      "answer":     "Concise complete answer (1-2 sentences)",
      "topic":      "Topic name",
      "difficulty": "easy | medium | hard"
    }}
  ],
  "quiz": [
    {{
      "question":       "Multiple choice question",
      "options":        {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "correct_answer": "A",
      "explanation":    "Why this answer is correct",
      "topic":          "Topic name"
    }}
  ]
}}

Generate 8-12 flashcards and 5-7 quiz questions. Keep flashcard answers to 1-2 sentences. Keep quiz explanations STRICTLY to 2 lines maximum. Keep everything concise to fit in the response."""


def prompt_audio_topics(transcript: str) -> str:
    """Extract structured knowledge from a lecture audio transcription."""
    excerpt        = transcript[:6000]
    truncated_note = "\n[... transcript truncated for length ...]" if len(transcript) > 6000 else ""

    return f"""You are an academic knowledge extraction assistant processing a lecture transcription.

=== LECTURE TRANSCRIPT ===
{excerpt}{truncated_note}

Instructions:
- Identify the main topics and concepts covered in this lecture.
- Extract key academic concepts with concise explanations.
- List the most important points a student should remember.
- Suggest relevant textbooks or topics for further reading.

Return ONLY a single valid JSON object. No markdown fences, no explanation:

{{
  "lecture_title":     "Inferred title of the lecture",
  "subject_area":      "e.g. Thermodynamics, Machine Learning, Economics",
  "topics_covered":    ["Topic 1", "Topic 2", "Topic 3"],
  "key_concepts":      [{{"concept": "concept name", "explanation": "clear explanation"}}],
  "important_points":  ["The most important things a student must remember"],
  "summary":           "A concise 3-5 sentence paragraph summarising the lecture content.",
  "suggested_reading": ["Book Title by Author", "Topic to research further"]
}}"""


# ──────────────────────────────────────────────────────────────────────────────
#  LECTURE-LEVEL SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

def prompt_lecture_summary(
    video_path:     str,
    frame_analyses: List[Dict[str, Any]],
    transcript:     str,
    sample_n:       int = 10,
) -> str:
    """Build a high-level lecture summary from sampled frame data and transcript."""
    step      = max(1, len(frame_analyses) // sample_n)
    snapshots = []
    for fr in frame_analyses[::step]:
        ac = fr.get("academic_content", {})
        snapshots.append(
            f"  [{fr['timestamp']:.1f}s] "
            f"slide='{ac.get('slide_title', '?')}' "
            f"concepts={ac.get('key_concepts', [])[:3]}"
        )

    return f"""You are generating a structured lecture summary for a student.

Video file    : {Path(video_path).name}
Total frames  : {len(frame_analyses)}

=== SLIDE TIMELINE (sampled key frames) ===
{chr(10).join(snapshots) or "  (no frames available)"}

=== TRANSCRIPT EXCERPT ===
{transcript[:2500] or "(no transcript available)"}

Instructions:
- Identify the overall lecture title and subject.
- List the main topics in order of appearance.
- Write clear learning outcomes a student should achieve.
- Assess the difficulty level honestly.

Return ONLY a single valid JSON object. No markdown fences, no explanation:

{{
  "lecture_title":     "Full descriptive title of this lecture",
  "subject_area":      "Academic subject area",
  "main_topics":       ["Topic 1", "Topic 2", "Topic 3"],
  "learning_outcomes": [
    "After this lecture, students will be able to explain ...",
    "Students will understand ...",
    "Students will be able to apply ..."
  ],
  "summary":           "Comprehensive 4-6 sentence paragraph summarising the entire lecture.",
  "difficulty_level":  "beginner | intermediate | advanced"
}}"""


# ──────────────────────────────────────────────────────────────────────────────
#  STUDY NOTES  (Markdown output — passed to reason_text())
# ──────────────────────────────────────────────────────────────────────────────

def prompt_study_notes(
    video_path:       str,
    frame_analyses:   List[Dict[str, Any]],
    transcript:       str,
    audio_topics:     Dict[str, Any],
    lecture_summary:  Dict[str, Any],
    sample_n:         int = 15,
    # v3: pre-deduplicated inputs (preferred when available)
    deduped_concepts: Optional[List[str]]      = None,
    deduped_formulas: Optional[List[str]]      = None,
    deduped_defs:     Optional[List[Dict]]     = None,
) -> str:
    """
    Build the prompt for generating comprehensive Markdown study notes.

    v3: When deduped_* parameters are provided they replace the raw
    per-frame data so the LLM receives clean, non-repetitive inputs.
    """
    # ── concepts ──────────────────────────────────────────────────────────────
    if deduped_concepts is not None:
        concepts = deduped_concepts[:40]
    else:
        step    = max(1, len(frame_analyses) // sample_n)
        sampled = frame_analyses[::step]
        concepts = list({
            c for fr in sampled
            for c in fr.get("academic_content", {}).get("key_concepts", [])
        })[:40]

    # ── bullet points (always from frames — not deduplicated) ─────────────────
    step    = max(1, len(frame_analyses) // sample_n)
    sampled = frame_analyses[::step]
    points  = [
        p for fr in sampled
        for p in fr.get("academic_content", {}).get("bullet_points", [])
    ][:30]

    # ── definitions ───────────────────────────────────────────────────────────
    if deduped_defs is not None:
        defs = deduped_defs[:20]
    else:
        defs = [
            d for fr in sampled
            for d in fr.get("academic_content", {}).get("definitions", [])
        ][:20]

    # ── formulas ──────────────────────────────────────────────────────────────
    if deduped_formulas is not None:
        formulas = deduped_formulas[:15]
    else:
        formulas = list({
            f for fr in sampled
            for f in fr.get("academic_content", {}).get("formulas", [])
        })[:15]

    return f"""You are an expert academic note-taker. Your job is to create comprehensive, well-structured Markdown study notes for a student based on the data below from a lecture video.

Lecture Title : {lecture_summary.get('lecture_title', Path(video_path).stem)}
Subject Area  : {lecture_summary.get('subject_area', 'General')}
Difficulty    : {lecture_summary.get('difficulty_level', 'unknown')}
Video File    : {Path(video_path).name}

=== LECTURE SUMMARY ===
{lecture_summary.get('summary', '(not available)')}

=== TOPICS COVERED (from audio) ===
{json.dumps(audio_topics.get('topics_covered', []), indent=2)}

=== LEARNING OUTCOMES ===
{json.dumps(lecture_summary.get('learning_outcomes', []), indent=2)}

=== KEY CONCEPTS (deduplicated) ===
{json.dumps(concepts, indent=2)}

=== IMPORTANT POINTS (from slides) ===
{json.dumps(points, indent=2)}

=== DEFINITIONS (deduplicated) ===
{json.dumps(defs, indent=2)}

=== FORMULAS (deduplicated) ===
{json.dumps(formulas, indent=2)}

=== TRANSCRIPT EXCERPT ===
{transcript[:3000] or "(no transcript)"}

Instructions:
Write comprehensive Markdown study notes. Include ALL of the following sections:
1. An overview paragraph
2. Learning objectives (bulleted)
3. Key concepts explained clearly
4. Definitions (as a definition list or table)
5. Formulas with brief explanations (if any)
6. Detailed notes on each major topic
7. A summary / conclusion paragraph
8. Review questions (5-7 questions a student can use for self-testing)

Make the notes detailed, educational, and easy for a student to read and revise from.
Use clear Markdown formatting: headers, bullet points, bold for key terms.

IMPORTANT: Return ONLY the Markdown text. Do NOT wrap it in JSON. Do NOT add any preamble.
Start immediately with:
# Study Notes: {lecture_summary.get('lecture_title', Path(video_path).stem)}"""


# ──────────────────────────────────────────────────────────────────────────────
#  FLASHCARDS
# ──────────────────────────────────────────────────────────────────────────────

def prompt_flashcards(
    video_path:       str,
    frame_analyses:   List[Dict[str, Any]],
    audio_topics:     Dict[str, Any],
    lecture_summary:  Dict[str, Any],
    deduped_concepts: Optional[List[str]]  = None,
    deduped_formulas: Optional[List[str]]  = None,
    deduped_defs:     Optional[List[Dict]] = None,
    # Caps to keep the prompt within Phi-3-mini's 4096-token context window.
    # The token budget after system + output overhead is ~2000 tokens for input.
    max_concepts: int = 15,
    max_defs:     int = 8,
    max_formulas: int = 8,
) -> str:
    """
    Generate Q&A flashcards from frame data and audio topics.

    v3: Accepts deduped_* inputs to avoid duplicate/repetitive cards.
    max_concepts/max_defs/max_formulas cap prompt length for small-context models.
    """
    # ── concepts ──────────────────────────────────────────────────────────────
    if deduped_concepts is not None:
        concepts = deduped_concepts[:max_concepts]
    else:
        concepts = list({
            c for fr in frame_analyses
            for c in fr.get("academic_content", {}).get("key_concepts", [])
        })[:max_concepts]

    # ── definitions ───────────────────────────────────────────────────────────
    if deduped_defs is not None:
        defs = deduped_defs[:max_defs]
    else:
        defs = [
            d for fr in frame_analyses
            for d in fr.get("academic_content", {}).get("definitions", [])
        ][:max_defs]

    # ── formulas ──────────────────────────────────────────────────────────────
    if deduped_formulas is not None:
        formulas = deduped_formulas[:max_formulas]
    else:
        formulas = list({
            f for fr in frame_analyses
            for f in fr.get("academic_content", {}).get("formulas", [])
        })[:max_formulas]

    # Cap audio inputs too — important_points can be verbose
    audio_concepts = audio_topics.get("key_concepts", [])[:8]
    important_pts  = audio_topics.get("important_points", [])[:5]

    return f"""You are an expert academic flashcard creator.

Lecture: {lecture_summary.get('lecture_title', Path(video_path).name)}
Subject: {lecture_summary.get('subject_area', 'General')}

=== CONCEPTS FROM SLIDES (deduplicated) ===
{json.dumps(concepts, indent=2)}

=== DEFINITIONS FROM SLIDES (deduplicated) ===
{json.dumps(defs, indent=2)}

=== FORMULAS FROM SLIDES (deduplicated) ===
{json.dumps(formulas, indent=2)}

=== KEY CONCEPTS FROM AUDIO ===
{json.dumps(audio_concepts, indent=2)}

=== IMPORTANT POINTS FROM AUDIO ===
{json.dumps(important_pts, indent=2)}

Instructions:
- Create 15 to 20 high-quality question-and-answer flashcards.
- Each card must test exactly ONE clearly defined concept, definition, or formula.
- Questions should be specific and unambiguous.
- Answers should be concise but complete (1-3 sentences).
- Cover a variety of difficulty levels (recall, comprehension, application).
- Assign each card a topic label.
- Do NOT create duplicate cards about the same concept.

Return ONLY a valid JSON array. No markdown fences, no explanation:

[
  {{
    "question": "What is ...",
    "answer":   "...",
    "topic":    "Topic name",
    "difficulty": "easy | medium | hard"
  }},
  ...
]"""


# ──────────────────────────────────────────────────────────────────────────────
#  QUIZ
# ──────────────────────────────────────────────────────────────────────────────

def prompt_quiz(
    lecture_summary: Dict[str, Any],
    audio_topics:    Dict[str, Any],
    frame_analyses:  List[Dict[str, Any]],
) -> str:
    """Generate a multiple-choice quiz from lecture content."""
    concepts = list({
        c for fr in frame_analyses
        for c in fr.get("academic_content", {}).get("key_concepts", [])
    })[:20]

    return f"""You are an academic quiz generator.

Lecture: {lecture_summary.get('lecture_title', 'Lecture')}
Subject: {lecture_summary.get('subject_area', 'General')}
Topics : {json.dumps(audio_topics.get('topics_covered', []))}
Concepts: {json.dumps(concepts)}

Create a 10-question multiple choice quiz covering the key content of this lecture.
Each question should have 4 options (A, B, C, D) with exactly one correct answer.
Explanations for the correct answer MUST be strictly 2 lines or less.

Return ONLY a valid JSON array. No markdown fences:

[
  {{
    "question":       "Question text?",
    "options":        {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct_answer": "A",
    "explanation":    "Why A is correct.",
    "topic":          "Topic name"
  }},
  ...
]"""


# ──────────────────────────────────────────────────────────────────────────────
#  SINGLE-CALL: everything in one Phi-3 generation  (kept for reference)
# ──────────────────────────────────────────────────────────────────────────────

def prompt_everything(
    video_path:     str,
    frame_analyses: List[Dict[str, Any]],
    transcript:     str,
    sample_n:       int = 8,
    max_concepts:   int = 10,
) -> str:
    """
    One LLM call → all Phase 2 outputs.

    NOTE (v4): This function is kept for reference but the pipeline now uses
    the two-call split (prompt_analysis_and_notes + prompt_flashcards_and_quiz)
    which reliably fits within Phi-3-mini's 4 096-token context window.

    This single-call version stalls on RTX 3050 because the prompt alone
    consumes ~2 800 tokens, leaving only ~1 200 tokens for all outputs
    combined — not enough for study_notes + flashcards + quiz together.
    """
    excerpt  = transcript[:2500]
    tr_note  = "\n[... truncated ...]" if len(transcript) > 2500 else ""

    step     = max(1, len(frame_analyses) // sample_n)
    timeline = []
    concepts_seen: List[str] = []
    for fr in frame_analyses[::step]:
        ac    = fr.get("academic_content", {})
        title = ac.get("slide_title", "")
        kc    = ac.get("key_concepts", [])[:2]
        concepts_seen.extend(kc)
        if title or kc:
            timeline.append(f"  [{fr['timestamp']:.0f}s] {title} {kc}")

    seen: set = set()
    unique_concepts: List[str] = []
    for c in concepts_seen:
        cl = c.lower().strip()
        if cl and cl not in seen:
            seen.add(cl)
            unique_concepts.append(c)
            if len(unique_concepts) >= max_concepts:
                break

    return f"""You are an academic content extraction and study material creation assistant.

Analyse this lecture and produce ALL outputs in one JSON response.

=== TRANSCRIPT (excerpt) ===
{excerpt}{tr_note}

=== SLIDE TIMELINE ===
{chr(10).join(timeline) or "  (animated video — extract from transcript)"}

=== CONCEPTS FROM SLIDES ===
{json.dumps(unique_concepts) if unique_concepts else "[]"}

Return ONLY this JSON object. No markdown fences, no preamble, nothing outside the JSON.
Keep values SHORT and CONCISE — every token counts:

{{
  "lecture_title":     "Title (max 8 words)",
  "subject_area":      "Subject",
  "difficulty":        "beginner | intermediate | advanced",
  "topics":            ["Topic 1", "Topic 2", "Topic 3"],
  "key_concepts":      ["concept 1", "concept 2", "concept 3", "concept 4"],
  "learning_outcomes": ["Outcome 1", "Outcome 2"],
  "summary":           "2-3 sentences only.",
  "flashcards": [
    {{"q": "Question?", "a": "Short answer.", "topic": "Topic"}},
    {{"q": "Question?", "a": "Short answer.", "topic": "Topic"}},
    {{"q": "Question?", "a": "Short answer.", "topic": "Topic"}},
    {{"q": "Question?", "a": "Short answer.", "topic": "Topic"}},
    {{"q": "Question?", "a": "Short answer.", "topic": "Topic"}}
  ],
  "quiz": [
    {{"q": "Question?", "A": "opt A", "B": "opt B", "C": "opt C", "D": "opt D", "ans": "A", "why": "Reason (max 2 lines)."}},
    {{"q": "Question?", "A": "opt A", "B": "opt B", "C": "opt C", "D": "opt D", "ans": "B", "why": "Reason (max 2 lines)."}},
    {{"q": "Question?", "A": "opt A", "B": "opt B", "C": "opt C", "D": "opt D", "ans": "C", "why": "Reason (max 2 lines)."}}
  ],
  "study_notes": "## Key Points\\n- Point about main concept 1\\n- Point about main concept 2\\n- Point about main concept 3\\n\\n## Summary\\nOne short paragraph summarising the lecture. Keep under 80 words."
}}"""


# ──────────────────────────────────────────────────────────────────────────────
#  TWO-CALL SPLIT  (v4 — replaces prompt_everything in the Phase 2 hot path)
# ──────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  CALL 1 — metadata only
#  Uses: llm.reason()  →  dict
#
#  Prompt is intentionally tiny:
#    - transcript capped at 1 200 chars  (~300 tok)
#    - timeline capped at 5 rows         (~50 tok)
#    - schema                            (~120 tok)
#  Total input: ~470 tok  →  output budget ~3 560 tok  (only ~250 needed)
# ─────────────────────────────────────────────────────────────────────────────
 
def prompt_metadata(
    video_path:     str,
    frame_analyses: List[Dict[str, Any]],
    transcript:     str,
    sample_n:       int = 5,
    max_concepts:   int = 6,
) -> str:
    """
    Call 1 of 3.
 
    Produces compact lecture metadata — nothing long, no notes, no cards.
    Keeping the output small guarantees json.loads() never sees a truncated
    object even on the smallest token budget.
 
    Output schema
    -------------
    {
      "lecture_title":     str,          # max 8 words
      "subject_area":      str,
      "difficulty":        str,          # beginner | intermediate | advanced
      "topics":            [str, ...],   # max 4
      "key_concepts":      [str, ...],   # max 6  ← fed into Call 2 + Call 3
      "learning_outcomes": [str, ...],   # max 3
      "summary":           str           # 2 sentences max
    }
    """
    # ── transcript: 1 200 chars ≈ 300 tokens ─────────────────────────────────
    excerpt = transcript[:1200]
    tr_note = " [truncated]" if len(transcript) > 1200 else ""
 
    # ── timeline: 5 rows ≈ 50 tokens ─────────────────────────────────────────
    step     = max(1, len(frame_analyses) // sample_n)
    timeline = []
    seen_c:   set        = set()
    concepts: List[str]  = []
 
    for fr in frame_analyses[::step]:
        ac    = fr.get("academic_content", {})
        title = ac.get("slide_title", "").strip()
        kc    = [c for c in ac.get("key_concepts", [])[:2] if c]
        for c in kc:
            cl = c.lower().strip()
            if cl and cl not in seen_c:
                seen_c.add(cl)
                concepts.append(c)
        if title or kc:
            timeline.append(f"[{fr['timestamp']:.0f}s] {title} {kc}")
        if len(timeline) >= sample_n:
            break
 
    concepts = concepts[:max_concepts]
 
    return (
        "Extract lecture metadata. Return ONLY this JSON, no markdown, no extra text:\n\n"
        f"TRANSCRIPT ({len(excerpt)} chars{tr_note}):\n{excerpt}\n\n"
        f"SLIDES: {' | '.join(timeline) or '(none)'}\n"
        f"SLIDE CONCEPTS: {json.dumps(concepts)}\n\n"
        '{\n'
        '  "lecture_title":     "Short title (max 8 words)",\n'
        '  "subject_area":      "Subject name",\n'
        '  "difficulty":        "beginner | intermediate | advanced",\n'
        '  "topics":            ["Topic 1", "Topic 2", "Topic 3"],\n'
        '  "key_concepts":      ["concept 1", "concept 2", "concept 3"],\n'
        '  "learning_outcomes": ["Students will understand ...", "Students will be able to ..."],\n'
        '  "summary":           "2 sentence summary."\n'
        '}'
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  CALL 2 — study notes as plain Markdown
#  Uses: llm.reason_text()  →  str
#
#  NO JSON — reason_text() returns raw text.
#  Truncation is safe: partial Markdown is still readable.
#
#  Prompt: ~500 tok  →  output budget ~3 500 tok  (we use ~500-600)
# ─────────────────────────────────────────────────────────────────────────────
 
def prompt_study_notes_text(
    lecture_title:    str,
    subject_area:     str,
    difficulty:       str,
    topics:           List[str],
    key_concepts:     List[str],
    learning_outcomes: List[str],
    summary:          str,
    formulas:         Optional[List[str]] = None,
    max_concepts:     int = 6,
    max_formulas:     int = 4,
) -> str:
    """
    Call 2 of 3.
 
    Plain-text Markdown output — no JSON escaping, no corruption risk.
    reason_text() is used instead of reason() so the output is returned
    as-is without any JSON parsing attempt.
 
    If generation truncates mid-sentence the notes are still usable.
    The study_notes field in the result dict is populated from this output.
    """
    concepts = key_concepts[:max_concepts]
    fmls     = (formulas or [])[:max_formulas]
    topics_s = ", ".join(topics[:4]) or "see summary"
 
    formula_block = (
        "\nFORMULAS: " + " | ".join(fmls) + "\n"
        if fmls else ""
    )
 
    return (
        f"Write concise Markdown study notes for a student.\n\n"
        f"LECTURE : {lecture_title or 'Lecture'}\n"
        f"SUBJECT : {subject_area or 'General'}\n"
        f"LEVEL   : {difficulty or 'unknown'}\n"
        f"TOPICS  : {topics_s}\n"
        f"SUMMARY : {summary or '(not available)'}\n"
        f"CONCEPTS: {', '.join(concepts) or '(none)'}\n"
        f"OUTCOMES: {'; '.join(learning_outcomes[:3]) or '(none)'}"
        f"{formula_block}\n\n"
        "Write the notes below. Use these sections (keep each section SHORT):\n"
        "1. ## Overview  (2-3 sentences)\n"
        "2. ## Key Concepts  (bullet list, one line each)\n"
        "3. ## Topics Covered  (bullet list)\n"
        "4. ## Learning Outcomes  (bullet list)\n"
        "5. ## Summary  (2-3 sentences)\n\n"
        "Use **bold** for key terms. Keep total notes under 400 words.\n\n"
        f"# Study Notes: {lecture_title or 'Lecture'}\n"
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  CALL 3 — flashcards + quiz
#  Uses: llm.reason()  →  dict
#
#  Input is only distilled fields from Call 1 — no raw frames, no transcript.
#  Prompt: ~350 tok  →  output budget ~3 700 tok  (we use ~550)
#
#  Reduced targets vs old functions: 6 cards + 4 MCQs
#  This keeps generation time under ~30s on RTX 3050 and avoids timeout.
# ─────────────────────────────────────────────────────────────────────────────
 
def prompt_cards_and_quiz(
    lecture_title:    str,
    subject_area:     str,
    key_concepts:     List[str],
    learning_outcomes: List[str],
    topics:           List[str],
    formulas:         Optional[List[str]] = None,
    max_concepts:     int = 6,
    max_outcomes:     int = 3,
    max_formulas:     int = 3,
) -> str:
    """
    Call 3 of 3.
 
    Generates 6 flashcards + 4 MCQ questions.
    Targets are intentionally modest (was 8+5) to keep generation time
    under ~30 seconds on RTX 3050 and guarantee the JSON closes cleanly.
 
    Output schema
    -------------
    {
      "flashcards": [
        {"question": str, "answer": str, "topic": str, "difficulty": str},
        ...                                              # 6 items
      ],
      "quiz": [
        {"question": str, "options": {"A":str,...,"D":str},
         "correct_answer": str, "explanation": str, "topic": str},
        ...                                              # 4 items
      ]
    }
    """
    concepts = key_concepts[:max_concepts]
    outcomes = learning_outcomes[:max_outcomes]
    fmls     = (formulas or [])[:max_formulas]
 
    formula_line = (
        f"\nFORMULAS: {json.dumps(fmls)}"
        if fmls else ""
    )
 
    return (
        f"Create flashcards and a quiz. Return ONLY this JSON, no markdown:\n\n"
        f"LECTURE : {lecture_title or 'Lecture'}\n"
        f"SUBJECT : {subject_area or 'General'}\n"
        f"TOPICS  : {', '.join(topics[:3]) or 'see concepts'}\n"
        f"CONCEPTS: {json.dumps(concepts)}\n"
        f"OUTCOMES: {json.dumps(outcomes)}"
        f"{formula_line}\n\n"
        '{\n'
        '  "flashcards": [\n'
        '    {"question": "What is ...?", "answer": "...", "topic": "...", "difficulty": "easy | medium | hard"},\n'
        '    {"question": "...", "answer": "...", "topic": "...", "difficulty": "..."}\n'
        '  ],\n'
        '  "quiz": [\n'
        '    {"question": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, '
        '"correct_answer": "A", "explanation": "...", "topic": "..."},\n'
        '    {"question": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, '
        '"correct_answer": "B", "explanation": "...", "topic": "..."}\n'
        '  ]\n'
        '}\n\n'
        'Generate exactly 6 flashcards and 4 quiz questions. '
        'Keep flashcard answers SHORT (1 sentence). '
        'Keep quiz explanations STRICTLY to 2 lines maximum. Do not add extra keys.'
    )