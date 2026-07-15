"""
academic_system/pdf_generator.py
===================================
Generates a comprehensive multi-section academic PDF report using
ReportLab Platypus.

PDF Sections:
  1. Cover page          — title, subject, difficulty, stats
  2. Lecture Overview    — summary, topics, learning outcomes
  3. Key Concepts        — audio-derived concepts + table of definitions
  4. Formulas            — all unique formulas detected across frames
  5. Slide-by-Slide      — high/medium importance frames with timestamps
  6. Transcript Excerpt  — formatted text from Whisper output
  7. Flashcards          — full Q&A review section
  8. Suggested Reading   — extracted from audio analysis

Usage:
    from academic_system.pdf_generator import generate_pdf_report
    pdf_path = generate_pdf_report(
        video_path=..., pdf_dir=..., lecture_summary=...,
        audio_topics=..., frame_analyses=...,
        flashcards=..., transcript_text=...,
    )
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _safe_xml(text: str) -> str:
    """Escape characters that break ReportLab Paragraph XML parsing."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def generate_pdf_report(
    video_path:      str,
    pdf_dir:         str,
    lecture_summary: Dict[str, Any],
    audio_topics:    Dict[str, Any],
    frame_analyses:  List[Dict[str, Any]],
    flashcards:      List[Dict[str, Any]],
    transcript_text: str,
) -> str:
    """
    Build and save the academic PDF report.

    Args:
        video_path:      Path to the original video file (used for naming).
        pdf_dir:         Directory where the PDF will be saved.
        lecture_summary: Dict from the lecture-level LLM call.
        audio_topics:    Dict from the audio-level LLM call.
        frame_analyses:  List of per-frame result dicts.
        flashcards:      List of {question, answer, topic, difficulty} dicts.
        transcript_text: Full Whisper transcript as a plain string.

    Returns:
        Absolute path to the saved PDF file.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak,
        Table, TableStyle, HRFlowable, KeepTogether,
    )

    # ── File path ──────────────────────────────────────────────────────────────
    stem     = Path(video_path).stem
    pdf_path = os.path.join(pdf_dir, f"{stem}_academic_report.pdf")
    os.makedirs(pdf_dir, exist_ok=True)

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
        title=lecture_summary.get("lecture_title", stem),
        author="Academic Intelligence System",
    )

    styles = getSampleStyleSheet()

    # ── Colour palette ─────────────────────────────────────────────────────────
    NAVY      = colors.HexColor("#1a3a5c")
    STEEL     = colors.HexColor("#2c5f8a")
    LIGHT_BG  = colors.HexColor("#f0f6ff")
    RULE_COL  = colors.HexColor("#ccddee")
    GREY_TXT  = colors.HexColor("#666666")
    CARD_BG   = colors.HexColor("#fffbea")

    # ── Paragraph styles ───────────────────────────────────────────────────────
    title_sty = ParagraphStyle(
        "AcadTitle", parent=styles["Title"],
        fontSize=24, spaceAfter=6, textColor=NAVY, leading=30,
    )
    subtitle_sty = ParagraphStyle(
        "AcadSubtitle", parent=styles["Normal"],
        fontSize=13, textColor=STEEL, spaceAfter=4, leading=18,
    )
    h1_sty = ParagraphStyle(
        "AcadH1", parent=styles["Heading1"],
        fontSize=15, spaceBefore=16, spaceAfter=4, textColor=NAVY,
    )
    h2_sty = ParagraphStyle(
        "AcadH2", parent=styles["Heading2"],
        fontSize=12, spaceBefore=10, spaceAfter=3, textColor=STEEL,
    )
    body_sty = ParagraphStyle(
        "AcadBody", parent=styles["Normal"],
        fontSize=10, leading=15, spaceAfter=4,
    )
    bull_sty = ParagraphStyle(
        "AcadBull", parent=styles["Normal"],
        fontSize=10, leading=14, leftIndent=14, spaceAfter=2,
    )
    cap_sty = ParagraphStyle(
        "AcadCap", parent=styles["Normal"],
        fontSize=8, textColor=GREY_TXT, spaceAfter=3, fontName="Helvetica-Oblique",
    )
    formula_sty = ParagraphStyle(
        "AcadFormula", parent=styles["Normal"],
        fontSize=11, leading=16, leftIndent=10,
        fontName="Courier-Bold", spaceAfter=4,
    )
    q_sty = ParagraphStyle(
        "CardQ", parent=styles["Normal"],
        fontSize=10, fontName="Helvetica-Bold", textColor=NAVY, leading=14,
    )
    a_sty = ParagraphStyle(
        "CardA", parent=styles["Normal"],
        fontSize=10, leading=14, leftIndent=10,
    )

    # ── Helper closures ────────────────────────────────────────────────────────

    def hr():
        return HRFlowable(width="100%", thickness=0.6, color=RULE_COL, spaceAfter=6)

    def p(text, sty=None):
        return Paragraph(_safe_xml(str(text)), sty or body_sty)

    def bullet(text):
        return Paragraph(f"• {_safe_xml(str(text))}", bull_sty)

    def section_header(num: str, title: str):
        return Paragraph(f"{num}. {_safe_xml(title)}", h1_sty)

    # ── Meta values ────────────────────────────────────────────────────────────
    lecture_title = (
        lecture_summary.get("lecture_title")
        or audio_topics.get("lecture_title")
        or stem
    )
    subject_area = (
        lecture_summary.get("subject_area")
        or audio_topics.get("subject_area")
        or "General"
    )
    difficulty = lecture_summary.get("difficulty_level", "").capitalize()

    story: list = []

    # ══════════════════════════════════════════════════════════════════════════
    #  COVER PAGE
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        Spacer(1, 2*cm),
        Paragraph("Academic Study Report", title_sty),
        hr(),
        Spacer(1, 0.3*cm),
        p(lecture_title, subtitle_sty),
        Spacer(1, 0.5*cm),
    ]

    cover_data = [
        ["Subject",        subject_area],
        ["Difficulty",     difficulty or "N/A"],
        ["Source file",    Path(video_path).name],
        ["Frames analysed", str(len(frame_analyses))],
        ["Flashcards",     str(len(flashcards))],
    ]
    cover_tbl = Table(cover_data, colWidths=[4*cm, 13*cm])
    cover_tbl.setStyle(TableStyle([
        ("FONTNAME",     (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 10),
        ("TEXTCOLOR",    (0, 0), (0, -1), NAVY),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LINEBELOW",    (0, 0), (-1, -2), 0.3, RULE_COL),
    ]))
    story += [cover_tbl, PageBreak()]

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 1: LECTURE OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    story += [section_header("1", "Lecture Overview"), hr()]

    overview = lecture_summary.get("summary") or audio_topics.get("summary") or ""
    if overview:
        story += [p(overview), Spacer(1, 0.3*cm)]

    # Topics
    main_topics = (
        lecture_summary.get("main_topics")
        or audio_topics.get("topics_covered")
        or []
    )
    if main_topics:
        story.append(p("Topics Covered", h2_sty))
        for t in main_topics:
            story.append(bullet(str(t)))
        story.append(Spacer(1, 0.2*cm))

    # Learning outcomes
    outcomes = lecture_summary.get("learning_outcomes") or []
    if outcomes:
        story.append(p("Learning Outcomes", h2_sty))
        for o in outcomes:
            story.append(bullet(str(o)))
        story.append(Spacer(1, 0.2*cm))

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 2: KEY CONCEPTS & DEFINITIONS
    # ══════════════════════════════════════════════════════════════════════════
    story += [section_header("2", "Key Concepts and Definitions"), hr()]

    # Concepts from audio
    ac_concepts = audio_topics.get("key_concepts") or []
    if ac_concepts:
        story.append(p("Core Concepts (from Lecture Audio)", h2_sty))
        for item in ac_concepts:
            if isinstance(item, dict):
                concept = item.get("concept", "")
                expl    = item.get("explanation", "")
                if concept:
                    story.append(p(f"<b>{_safe_xml(concept)}</b> — {_safe_xml(expl)}"))
            elif isinstance(item, str):
                story.append(bullet(item))
        story.append(Spacer(1, 0.2*cm))

    # Definitions table from OCR frames (deduplicated)
    seen_terms: set = set()
    def_rows = [
        [
            Paragraph("<b>Term</b>", body_sty),
            Paragraph("<b>Definition</b>", body_sty),
        ]
    ]
    for fr in frame_analyses:
        for d in fr.get("academic_content", {}).get("definitions", []):
            term = d.get("term", "").strip()
            defn = d.get("definition", "").strip()
            if term and term.lower() not in seen_terms:
                seen_terms.add(term.lower())
                def_rows.append([
                    Paragraph(_safe_xml(term), body_sty),
                    Paragraph(_safe_xml(defn), body_sty),
                ])

    if len(def_rows) > 1:
        story.append(p("Definitions from Slides", h2_sty))
        tbl = Table(def_rows, colWidths=[4.5*cm, 12*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT_BG, colors.white]),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("GRID",          (0, 0), (-1, -1), 0.3, RULE_COL),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ]))
        story += [tbl, Spacer(1, 0.3*cm)]

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 3: FORMULAS & EQUATIONS
    # ══════════════════════════════════════════════════════════════════════════
    seen_formulas: set = set()
    unique_formulas: List[str] = []
    for fr in frame_analyses:
        for f in fr.get("academic_content", {}).get("formulas", []):
            if f and f.strip() and f.strip() not in seen_formulas:
                seen_formulas.add(f.strip())
                unique_formulas.append(f.strip())

    if unique_formulas:
        story += [section_header("3", "Formulas and Equations"), hr()]
        for formula in unique_formulas:
            story.append(Paragraph(_safe_xml(formula), formula_sty))
        story.append(Spacer(1, 0.3*cm))
        next_section = "4"
    else:
        next_section = "3"

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 4: SLIDE-BY-SLIDE CONTENT
    # ══════════════════════════════════════════════════════════════════════════
    story += [section_header(next_section, "Slide-by-Slide Content"), hr()]

    notable = [
        fr for fr in frame_analyses
        if fr.get("academic_content", {}).get("importance") in ("high", "medium")
        and (
            fr.get("academic_content", {}).get("slide_title")
            or fr.get("academic_content", {}).get("key_concepts")
        )
    ][:35]  # cap at 35 slides in the PDF

    if notable:
        for fr in notable:
            ac          = fr.get("academic_content", {})
            slide_title = ac.get("slide_title") or f"Frame {fr['frame_id']}"
            ts          = fr.get("timestamp", 0.0)
            importance  = ac.get("importance", "")

            block = [
                p(f"<b>{_safe_xml(slide_title)}</b>", h2_sty),
                Paragraph(
                    f"<i>Timestamp: {ts:.1f}s  |  Importance: {importance}</i>",
                    cap_sty
                ),
            ]
            if ac.get("content_summary"):
                block.append(p(ac["content_summary"]))
            for pt in (ac.get("bullet_points") or [])[:6]:
                block.append(bullet(str(pt)))
            if ac.get("key_concepts"):
                block.append(
                    p(f"<i>Concepts: {', '.join(_safe_xml(c) for c in ac['key_concepts'])}</i>")
                )
            block.append(Spacer(1, 0.15*cm))

            try:
                story.append(KeepTogether(block))
            except Exception:
                story.extend(block)
    else:
        story.append(p("No slides were flagged as high or medium importance."))

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 5: TRANSCRIPT EXCERPT
    # ══════════════════════════════════════════════════════════════════════════
    if transcript_text and transcript_text.strip():
        story += [PageBreak(), section_header(str(int(next_section)+1), "Audio Transcript (Excerpt)"), hr()]

        # Split into readable paragraphs at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', transcript_text)
        chunk, chunks = "", []
        for s in sentences:
            if len(chunk) + len(s) < 700:
                chunk = (chunk + " " + s).strip()
            else:
                if chunk:
                    chunks.append(chunk)
                chunk = s
        if chunk:
            chunks.append(chunk)

        for para_text in chunks[:15]:  # ~10,000 chars
            story += [p(para_text), Spacer(1, 0.08*cm)]

        if len(transcript_text) > 10000:
            story.append(Paragraph(
                "<i>Transcript continues — retrieve full text via GET /results/audio</i>",
                cap_sty
            ))

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 6: FLASHCARDS
    # ══════════════════════════════════════════════════════════════════════════
    if flashcards:
        story += [PageBreak(), section_header(str(int(next_section)+2), "Flashcards — Q&amp;A Review"), hr()]

        for i, card in enumerate(flashcards, 1):
            topic      = card.get("topic",      "")
            question   = card.get("question",   "")
            answer     = card.get("answer",     "")
            difficulty = card.get("difficulty", "")

            meta_parts = []
            if topic:
                meta_parts.append(topic)
            if difficulty:
                meta_parts.append(difficulty)

            card_block = []
            if meta_parts:
                card_block.append(
                    Paragraph(" | ".join(_safe_xml(m) for m in meta_parts), cap_sty)
                )
            card_block += [
                Paragraph(f"Q{i}: {_safe_xml(question)}", q_sty),
                Paragraph(f"A: {_safe_xml(answer)}", a_sty),
                Spacer(1, 0.2*cm),
            ]
            try:
                story.append(KeepTogether(card_block))
            except Exception:
                story.extend(card_block)

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 7: SUGGESTED READING
    # ══════════════════════════════════════════════════════════════════════════
    suggested = audio_topics.get("suggested_reading") or []
    if suggested:
        story += [
            section_header(str(int(next_section)+3), "Suggested Reading and Resources"),
            hr(),
        ]
        for item in suggested:
            story.append(bullet(str(item)))

    # ── Build PDF ──────────────────────────────────────────────────────────────
    doc.build(story)
    logger.info(f"[PDF] Report saved → {pdf_path}")
    return pdf_path