"""
academic_system/prompts.py
============================
All LLM prompt-builder functions for the Academic Intelligence System.

v3.0.3 addition
----------------
- Added prompt_cards_from_notes(): used by the new POST /generate/flashcards/{stem}
  endpoint to generate flashcards + quiz on demand AFTER the pipeline completes.

  Input combines TWO sources:
    ① Study-notes Markdown (already saved to disk / shown to the student)
       → gives the LLM clean, human-readable content to base questions on.
    ② Raw concepts + transcript excerpt from pipeline's in-memory data
       → richer signal for varied question topics (formulas, audio content).

  Prompt is compact (~500 tok) leaving ~700 tok for 6 flashcards + 4 MCQs —
  comfortably within Phi-3-mini's 4 096-token context window on RTX 3050.

v4 changes (2-call split fix)
------------------------------
- Added prompt_analysis_and_notes(): Call 1 of the new 2-call Phase 2 path.
- Added prompt_flashcards_and_quiz(): Call 2 of the new 2-call Phase 2 path.
- WHY the original prompt_everything() stalled: see module docstring there.

v3 changes
----------
- prompt_study_notes() and prompt_flashcards() now accept optional
  deduped_concepts, deduped_formulas, deduped_defs parameters.
- All functions remain fully backwards-compatible.
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
      "flashcards"  — list of Q&A cards
      "quiz"        — list of MCQ questions
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
    deduped_concepts: Optional[List[str]]      = None,
    deduped_formulas: Optional[List[str]]      = None,
    deduped_defs:     Optional[List[Dict]]     = None,
) -> str:
    """Build the prompt for generating comprehensive Markdown study notes."""
    if deduped_concepts is not None:
        concepts = deduped_concepts[:40]
    else:
        step    = max(1, len(frame_analyses) // sample_n)
        sampled = frame_analyses[::step]
        concepts = list({
            c for fr in sampled
            for c in fr.get("academic_content", {}).get("key_concepts", [])
        })[:40]

    step    = max(1, len(frame_analyses) // sample_n)
    sampled = frame_analyses[::step]
    points  = [
        p for fr in sampled
        for p in fr.get("academic_content", {}).get("bullet_points", [])
    ][:30]

    if deduped_defs is not None:
        defs = deduped_defs[:20]
    else:
        defs = [
            d for fr in sampled
            for d in fr.get("academic_content", {}).get("definitions", [])
        ][:20]

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
    max_concepts: int = 15,
    max_defs:     int = 8,
    max_formulas: int = 8,
) -> str:
    """Generate Q&A flashcards from frame data and audio topics."""
    if deduped_concepts is not None:
        concepts = deduped_concepts[:max_concepts]
    else:
        concepts = list({
            c for fr in frame_analyses
            for c in fr.get("academic_content", {}).get("key_concepts", [])
        })[:max_concepts]

    if deduped_defs is not None:
        defs = deduped_defs[:max_defs]
    else:
        defs = [
            d for fr in frame_analyses
            for d in fr.get("academic_content", {}).get("definitions", [])
        ][:max_defs]

    if deduped_formulas is not None:
        formulas = deduped_formulas[:max_formulas]
    else:
        formulas = list({
            f for fr in frame_analyses
            for f in fr.get("academic_content", {}).get("formulas", [])
        })[:max_formulas]

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

    NOTE (v3.0.3): Kept for reference only.  The pipeline now uses the 2-call
    split (prompt_metadata + prompt_study_notes_text) and flashcards are
    generated on demand via prompt_cards_from_notes().
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
#  TWO-CALL SPLIT  (v3.0.2/v3.0.3 pipeline — Call 1 & Call 2)
# ──────────────────────────────────────────────────────────────────────────────

def prompt_metadata(
    video_path:     str,
    frame_analyses: List[Dict[str, Any]],
    transcript:     str,
    sample_n:       int = 5,
    max_concepts:   int = 6,
) -> str:
    """
    Pipeline Call 1 of 2.

    Produces compact lecture metadata — nothing long, no notes, no cards.

    Output schema
    -------------
    {
      "lecture_title":     str,
      "subject_area":      str,
      "difficulty":        str,
      "topics":            [str, ...],
      "key_concepts":      [str, ...],
      "learning_outcomes": [str, ...],
      "summary":           str
    }
    """
    excerpt = transcript[:1200]
    tr_note = " [truncated]" if len(transcript) > 1200 else ""

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
    Pipeline Call 2 of 2.

    Plain-text Markdown output — no JSON escaping, no corruption risk.
    reason_text() is used instead of reason() so the output is returned
    as-is without any JSON parsing attempt.

    If generation truncates mid-sentence the notes are still usable.
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
    Old pipeline Call 3 — kept for the combined-outputs fallback path.

    As of v3.0.3 this is no longer called from run_academic_pipeline().
    Use prompt_cards_from_notes() for the new on-demand endpoint instead.

    Generates 6 flashcards + 4 MCQ questions.
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


# ──────────────────────────────────────────────────────────────────────────────
#  ON-DEMAND FLASHCARD GENERATION  (v3.0.3 — new)
# ──────────────────────────────────────────────────────────────────────────────

def prompt_cards_from_notes(
    notes_md:          str,
    lecture_title:     str,
    subject_area:      str,
    key_concepts:      List[str],
    formulas:          List[str],
    transcript:        str,
    topics:            List[str],
    learning_outcomes: List[str],
    # Caps to keep within Phi-3-mini's 4 096-token context window
    max_notes_chars:   int = 500,
    max_transcript_chars: int = 200,
    max_concepts:      int = 5,
    max_formulas:      int = 3,
    max_topics:        int = 3,
    max_outcomes:      int = 2,
) -> str:
    """
    ON-DEMAND flashcard + quiz generation (v3.0.3).

    Called by POST /generate/flashcards/{stem} AFTER the pipeline has finished
    generating study notes.  Unlike the old Call 3, this function combines
    TWO input sources for richer question coverage:

    Source A — notes_md (study-notes Markdown)
        The exact text the student already sees.  The LLM reads the headers,
        bullet points, and explanations to form natural questions.  Capped at
        max_notes_chars characters to respect the context window.

    Source B — raw pipeline data (concepts, formulas, transcript excerpt)
        Provides additional signal that may not have made it into the notes
        verbatim: formula labels, topic taxonomy, audio content.  Each field
        is independently capped so the total prompt stays compact.

    Token budget (typical)
    ----------------------
    Prompt : ~480 tokens  (notes 200 + transcript 100 + schema + labels)
    Output : ~650 tokens  (6 flashcards ≈ 350 tok + 4 MCQs ≈ 300 tok)
    Total  : ~1 130 tokens  — well within Phi-3-mini's 4 096-token window.

    Output schema
    -------------
    {
      "flashcards": [
        {"question": str, "answer": str, "topic": str, "difficulty": str},
        ...  # 6 items
      ],
      "quiz": [
        {"question": str, "options": {"A":str,...,"D":str},
         "correct_answer": str, "explanation": str, "topic": str},
        ...  # 4 items
      ]
    }
    """
    # ── Truncate / cap each source ─────────────────────────────────────────────
    notes_excerpt      = notes_md.strip()[:max_notes_chars]
    notes_truncated    = " [truncated]" if len(notes_md) > max_notes_chars else ""

    transcript_excerpt = transcript.strip()[:max_transcript_chars]
    tr_truncated       = " [truncated]" if len(transcript) > max_transcript_chars else ""

    concepts = key_concepts[:max_concepts]
    fmls     = formulas[:max_formulas]
    topics_s = ", ".join(topics[:max_topics]) or "see notes"
    outcomes = learning_outcomes[:max_outcomes]

    formula_line = (
        f"\nFORMULAS : {' | '.join(fmls)}"
        if fmls else ""
    )

    return (
        "Create flashcards and a quiz from the lecture notes below.\n"
        "Return ONLY this JSON object — no markdown, no preamble:\n\n"

        # ── Source A: study notes ─────────────────────────────────────────────
        f"=== STUDY NOTES ({len(notes_excerpt)} chars{notes_truncated}) ===\n"
        f"{notes_excerpt}\n\n"

        # ── Source B: raw pipeline data ───────────────────────────────────────
        f"LECTURE  : {lecture_title or 'Lecture'}\n"
        f"SUBJECT  : {subject_area  or 'General'}\n"
        f"TOPICS   : {topics_s}\n"
        f"CONCEPTS : {json.dumps(concepts)}\n"
        f"OUTCOMES : {json.dumps(outcomes)}"
        f"{formula_line}\n\n"

        # ── Transcript excerpt (audio signal) ─────────────────────────────────
        + (
            f"TRANSCRIPT ({len(transcript_excerpt)} chars{tr_truncated}):\n"
            f"{transcript_excerpt}\n\n"
            if transcript_excerpt else ""
        )

        # ── Output schema ─────────────────────────────────────────────────────
        + '{\n'
          '  "flashcards": [\n'
          '    {"question": "What is ...?", "answer": "...", '
          '"topic": "...", "difficulty": "easy | medium | hard"},\n'
          '    {"question": "...", "answer": "...", '
          '"topic": "...", "difficulty": "..."}\n'
          '  ],\n'
          '  "quiz": [\n'
          '    {"question": "...", '
          '"options": {"A": "...", "B": "...", "C": "...", "D": "..."}, '
          '"correct_answer": "A", "explanation": "...", "topic": "..."},\n'
          '    {"question": "...", '
          '"options": {"A": "...", "B": "...", "C": "...", "D": "..."}, '
          '"correct_answer": "B", "explanation": "...", "topic": "..."}\n'
          '  ]\n'
          '}\n\n'

          # ── Generation instructions ───────────────────────────────────────
          'Rules:\n'
          '- Generate exactly 4 flashcards and 3 quiz questions.\n'
          '- Base questions on the study notes content above.\n'
          '- Use concepts, formulas, and transcript to add variety.\n'
          '- Each flashcard tests ONE concept. Answers: 1 sentence max.\n'
          '- Each quiz question has exactly one correct option (A–D).\n'
          '- Quiz explanations MUST be strictly 2 lines maximum.\n'
          '- Do NOT duplicate questions. Do NOT add extra JSON keys.\n'
          '- Keep the full JSON under 650 tokens so it fits the context window.'
    )