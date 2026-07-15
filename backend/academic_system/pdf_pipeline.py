"""
academic_system/pdf_pipeline.py
================================
PDF lecture-notes pipeline for AcademIQ v3.2.1 — PATCHED v2.

Fixes applied
-------------
  FIX 1 — Concurrent page analysis
    Pages processed in parallel via ThreadPoolExecutor (PDF_BATCH_SIZE workers,
    default 4, tunable via PDF_BATCH_SIZE env var).  Replaces the previous
    sequential `for i in range(n_pages)` loop.

  FIX 2 — Robust JSON fence stripping
    _robust_strip_fences() now calls raw.strip() BEFORE applying the ^-anchored
    regex so that LLM output with leading whitespace (e.g. '  ```json\\n{')
    is handled correctly.

  FIX 3 — Missing-comma repair between adjacent JSON objects
    _repair_json() now fixes `}\\n{` → `},\\n{` in addition to the
    existing `"value"\\n"key"` pattern.

  FIX 4 — _shared_llm write-back  ← THE CRASH FIX
    Root cause of the crash
    -----------------------
    run_pdf_pipeline() received shared_llm as a parameter and used it locally,
    but NEVER wrote it back to main_v7-9._shared_llm (the module-level global).

    After the PDF pipeline finished, main._shared_llm was still None.
    When the user called POST /generate/flashcards/{stem}, the check:

        if _shared_llm is not None:          # ← False for PDF uploads!
            await _run_flashcard_generation(...)
        else:
            await _run_flashcard_generation_with_llm(...)  # ← loads Phi-3 AGAIN

    fell into the else branch, which tried to load Phi-3 a second time while
    it was already occupying VRAM → OOM / crash on 4 GB GPUs.

    Compare with the video pipeline, where _load_models() explicitly does:
        global _shared_llm
        _shared_llm = llm          ← this is the line that was missing for PDF

    Fix
    ---
    After acquiring a valid llm instance (passed-in or freshly loaded),
    we call _register_shared_llm(llm) which locates main_v7-9 in sys.modules
    and sets its _shared_llm attribute directly.  This is safe to call at
    any point and non-fatal if the lookup fails.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# ── Optional PDF backends ─────────────────────────────────────────────────────
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from video_pipeline.config1 import config
from video_pipeline.detection.ocr import OCRExtractor
from video_pipeline.reasoning.phi3_engine import Phi3Reasoner as LlamaReasoner
from video_pipeline.utils.device import setup_device
from video_pipeline.utils.logger import get_logger

from academic_system.knowledge_graph import KnowledgeGraphBuilder
from academic_system.deduplicator import SemanticDeduplicator
from academic_system.language_support import LanguageDetector
from academic_system.pdf_generator import generate_pdf_report

from academic_system.prompts1 import (
    prompt_image_explain,
    prompt_image_batch_metadata,
    prompt_image_study_notes,
)

logger = get_logger(__name__)

# ── Re-use singletons from main (injected at runtime) ────────────────────────
_deduplicator:  Optional[SemanticDeduplicator]  = None
_graph_builder: Optional[KnowledgeGraphBuilder] = None
_lang_detector: Optional[LanguageDetector]      = None

NOTES_DIR     = "study_notes"
PDF_DIR       = "pdf_reports"
GRAPH_DIR     = "knowledge_graphs"
FLASHCARD_DIR = "flashcards"
IMAGE_DIR     = "images"

PAGE_RENDER_DPI:  int = int(os.environ.get("PDF_PAGE_DPI",         "150"))
MAX_PAGES_PHASE2: int = int(os.environ.get("PDF_MAX_PAGES_PHASE2",  "40"))
PDF_BATCH_SIZE:   int = int(os.environ.get("PDF_BATCH_SIZE",         "4"))


def init_pdf_pipeline_singletons(
    dedup: SemanticDeduplicator,
    graph: KnowledgeGraphBuilder,
    lang:  LanguageDetector,
) -> None:
    global _deduplicator, _graph_builder, _lang_detector
    _deduplicator  = dedup
    _graph_builder = graph
    _lang_detector = lang


def _check_pdf_backends() -> Dict[str, bool]:
    return {
        "pdfplumber": PDFPLUMBER_AVAILABLE,
        "pymupdf":    FITZ_AVAILABLE,
        "pillow":     PIL_AVAILABLE,
    }


# ── FIX 4: write the loaded LLM back to main's module-level global ────────────

def _register_shared_llm(llm: LlamaReasoner, shared_llm_ref: Optional[List] = None) -> None:
    """
    Write llm back into the mutable ref passed from main_v7-9.
    This avoids the fragile sys.modules name search entirely.
    Falls back to sys.modules search if no ref is passed (backward compatibility).
    """
    if shared_llm_ref is not None:
        try:
            shared_llm_ref[0] = llm
            logger.info("[PDF Pipeline] _shared_llm written back via direct ref.")
            return
        except Exception as exc:
            logger.error(f"[PDF Pipeline] Write to shared_llm_ref failed: {exc}")

    # Fallback to sys.modules lookup if ref not provided
    try:
        main_module = None
        for name, mod in sys.modules.items():
            if mod is None:
                continue
            # Match the main application module by known attributes
            if hasattr(mod, "_shared_llm") and hasattr(mod, "academic_results") and hasattr(mod, "pipeline_running"):
                main_module = mod
                break

        if main_module is not None:
            main_module._shared_llm = llm
            logger.info(
                f"[PDF Pipeline] _shared_llm registered in '{main_module.__name__}' — "
                "flashcard generation will reuse this LLM instance."
            )
        else:
            logger.warning(
                "[PDF Pipeline] Could not locate main module in sys.modules. "
                "_shared_llm NOT updated — flashcard POST may reload Phi-3."
            )
    except Exception as exc:
        logger.warning(f"[PDF Pipeline] _register_shared_llm fallback failed (non-fatal): {exc}")


# ── Robust JSON helpers (FIX 2 + FIX 3) ─────────────────────────────────────

def _robust_strip_fences(raw: str) -> str:
    """Strip markdown code fences regardless of leading/trailing whitespace."""
    if not raw:
        return raw
    # CRITICAL: strip outer whitespace FIRST so the ^ anchor works
    s = raw.strip()
    s = re.sub(r"^\s*```(?:json)?\s*\n?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\n?\s*```\s*$",          "", s, flags=re.IGNORECASE)
    return s.strip()


def _repair_json(s: str) -> str:
    """Repair common LLM JSON syntax errors."""
    if not s:
        return s

    # Known property-name typos
    typo_repairs = [
        (r'"questionimoine"\s*:', '"question":'),
        (r'"questin"\s*:',        '"question":'),
        (r'"questoin"\s*:',       '"question":'),
        (r'"answr"\s*:',          '"answer":'),
        (r'"answeer"\s*:',        '"answer":'),
        (r'"optons"\s*:',         '"options":'),
        (r'"correctanswer"\s*:',  '"correct_answer":'),
        (r'"correct_answeer"\s*:','"correct_answer":'),
        (r'"explnation"\s*:',     '"explanation":'),
        (r'"explantion"\s*:',     '"explanation":'),
    ]
    for pattern, replacement in typo_repairs:
        s = re.sub(pattern, replacement, s, flags=re.IGNORECASE)

    # FIX 3: missing comma between adjacent objects:  }\n{  →  },\n{
    s = re.sub(r'(\})\s*\n(\s*\{)', r'\1,\n\2', s)

    # Missing comma after string value before next key
    s = re.sub(r'(")\s*\n(\s*")',   r'\1,\n\2', s)

    return s


def _extract_first_json_object(raw: str) -> Dict:
    """Scan for the first complete {...} object in a string."""
    if not raw:
        return {}

    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$",           "", cleaned.strip())

    start = cleaned.find("{")
    if start == -1:
        return {}

    depth  = 0
    in_str = False
    escape = False

    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if escape:
            escape = False; continue
        if ch == "\\" and in_str:
            escape = True; continue
        if ch == '"' and not escape:
            in_str = not in_str; continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start: i + 1]
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


def _parse_llm_json(raw_text: str, context: str = "") -> Dict:
    """Multi-pass JSON parser used for metadata calls in the PDF pipeline."""
    if not raw_text:
        return {}

    cleaned  = _robust_strip_fences(raw_text)
    repaired = _repair_json(cleaned)

    try:
        result = json.loads(repaired)
        if isinstance(result, dict):
            logger.debug(f"[PDF JSON/{context}] Pass 1 (direct parse) OK.")
            return result
    except json.JSONDecodeError:
        pass

    result = _extract_first_json_object(repaired)
    if result:
        logger.info(f"[PDF JSON/{context}] Pass 2 (object scan) recovered result.")
        return result

    logger.warning(
        f"[PDF JSON/{context}] All parse passes failed. "
        f"Raw ({len(raw_text)} chars): {raw_text[:200]!r}"
    )
    return {}


# ── PDF file helpers ──────────────────────────────────────────────────────────

def _extract_text_per_page(pdf_path: str) -> List[str]:
    if not PDFPLUMBER_AVAILABLE:
        return []
    try:
        texts: List[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    raw = page.extract_text() or ""
                    texts.append(raw.strip())
                except Exception:
                    texts.append("")
        return texts
    except Exception:
        return []


def _render_pages_to_images(
    pdf_path:   str,
    output_dir: str,
    stem:       str,
    dpi:        int = PAGE_RENDER_DPI,
) -> List[str]:
    if not FITZ_AVAILABLE:
        return []
    os.makedirs(output_dir, exist_ok=True)
    paths: List[str] = []
    try:
        doc  = fitz.open(pdf_path)
        zoom = dpi / 72.0
        mat  = fitz.Matrix(zoom, zoom)
        for page_num in range(len(doc)):
            page  = doc.load_page(page_num)
            pix   = page.get_pixmap(matrix=mat, alpha=False)
            fname = f"{stem}_page_{page_num + 1:04d}.jpg"
            fpath = os.path.join(output_dir, fname)
            pix.save(fpath)
            paths.append(fpath)
        doc.close()
    except Exception as exc:
        logger.warning(f"[PDF render] {exc}")
    return paths


def _page_count(pdf_path: str) -> int:
    if FITZ_AVAILABLE:
        try:
            doc = fitz.open(pdf_path); n = len(doc); doc.close(); return n
        except Exception:
            pass
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return len(pdf.pages)
        except Exception:
            pass
    return 0


def _ocr_to_text(raw_ocr: Any) -> str:
    if isinstance(raw_ocr, list):
        parts = []
        for item in raw_ocr:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(p for p in parts if p).strip()
    return str(raw_ocr).strip()


# ── Per-page analysis (sync — runs inside thread pool) ───────────────────────

def analyse_pdf_page(
    page_image_path: str,
    pdfplumber_text: str,
    ocr:             OCRExtractor,
    llm:             LlamaReasoner,
    lang_info:       Optional[Dict] = None,
    page_num:        int = 1,
    total_pages:     int = 1,
) -> Dict[str, Any]:
    filename = (
        Path(page_image_path).name if page_image_path else f"page_{page_num}.jpg"
    )

    frame    = None
    ocr_text = ""
    if page_image_path and os.path.isfile(page_image_path):
        frame = cv2.imread(page_image_path)
    if frame is not None:
        try:
            raw_ocr  = ocr.extract(frame, config.ocr_confidence_threshold)
            ocr_text = _ocr_to_text(raw_ocr)
        except Exception:
            pass

    if len(pdfplumber_text) >= 40:
        combined_text = pdfplumber_text + (
            "\n\n" + ocr_text if ocr_text and len(ocr_text) > 40 else ""
        )
    else:
        combined_text = ocr_text or pdfplumber_text
    combined_text = combined_text.strip()

    academic_content: Dict[str, Any] = {}
    if combined_text and llm is not None:
        try:
            page_label = f"Page {page_num} of {total_pages}"
            img_prompt = prompt_image_explain(combined_text, page_num, page_label)
            if lang_info and _lang_detector:
                img_prompt = _lang_detector.patch_prompt(img_prompt, lang_info)
            result = llm.reason(img_prompt)
            if isinstance(result, dict):
                academic_content = result
        except Exception as exc:
            logger.warning(f"[PDF page {page_num}] LLM failed: {exc}")

    if not academic_content:
        academic_content = {
            "image_title":     f"Page {page_num}",
            "content_type":    "text",
            "importance":      "medium" if combined_text else "low",
            "description":     combined_text[:300] if combined_text else "No content detected.",
            "key_concepts":    [],
            "formulas":        [],
            "bullet_points":   [],
            "content_summary": combined_text[:200] if combined_text else "",
        }

    return {
        "frame_id":         page_num,
        "timestamp":        float(page_num),
        "filename":         filename,
        "frame_path":       os.path.relpath(page_image_path) if page_image_path else "",
        "frame_url":        "",
        "academic_content": academic_content,
        "ocr_text":         combined_text,
        "pdf_text":         pdfplumber_text,
        "image_title":      academic_content.get("image_title", f"Page {page_num}"),
        "description":      academic_content.get("description", ""),
        "content_type":     academic_content.get("content_type", "text"),
        "key_concepts":     academic_content.get("key_concepts", []),
        "bullet_points":    academic_content.get("bullet_points", []),
        "formulas":         academic_content.get("formulas", []),
        "content_summary":  academic_content.get("content_summary", ""),
    }


def _write_text_file(path: str, content: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _serialize(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


# ── FIX 1: concurrent page analysis ──────────────────────────────────────────

async def _analyse_pages_concurrent(
    n_pages:     int,
    image_paths: List[str],
    page_texts:  List[str],
    ocr:         OCRExtractor,
    llm:         LlamaReasoner,
    lang_info:   Optional[Dict],
    vr:          Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Dispatch all pages to a thread pool and collect results."""

    page_analyses: List[Optional[Dict]] = [None] * n_pages

    def _analyse_one(i: int):
        result = analyse_pdf_page(
            page_image_path = image_paths[i] if i < len(image_paths) else "",
            pdfplumber_text = page_texts[i]  if i < len(page_texts)  else "",
            ocr             = ocr,
            llm             = llm,
            lang_info       = lang_info,
            page_num        = i + 1,
            total_pages     = n_pages,
        )
        return i, result

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=PDF_BATCH_SIZE,
        thread_name_prefix="pdf_page",
    ) as pool:
        futures = {pool.submit(_analyse_one, i): i for i in range(n_pages)}
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                idx, result = future.result()
                page_analyses[idx] = result
                vr["per_frame_details"].append(result)
                done_count += 1
                if done_count % max(1, PDF_BATCH_SIZE) == 0 or done_count == n_pages:
                    logger.info(
                        f"[PDF Pipeline] Pages analysed: {done_count}/{n_pages}"
                    )
            except Exception as exc:
                logger.error(f"[PDF page worker] Failed: {exc}", exc_info=True)

    # Fill any None slots (failed pages) with a placeholder
    for i, p in enumerate(page_analyses):
        if p is None:
            page_analyses[i] = {
                "frame_id": i + 1, "timestamp": float(i + 1),
                "filename": f"page_{i + 1}.jpg", "frame_path": "",
                "frame_url": "", "academic_content": {"importance": "low"},
                "ocr_text": "", "pdf_text": "",
                "image_title": f"Page {i + 1}", "description": "",
                "content_type": "text", "key_concepts": [],
                "bullet_points": [], "formulas": [], "content_summary": "",
            }

    # Restore page order (as_completed is unordered)
    page_analyses.sort(key=lambda p: p["frame_id"])
    return page_analyses


# ── Main entry point ──────────────────────────────────────────────────────────

async def run_pdf_pipeline(
    pdf_path:             str,
    user_id:              int,
    batch_stem:           str,
    shared_llm:           Optional[LlamaReasoner],
    academic_results:     Dict[str, Any],
    flashcard_states:     Dict[str, Any],
    pipeline_running_ref: List[bool],
    shared_llm_ref:       Optional[List] = None,  # mutable ref for writing back loaded LLM
) -> None:
    pipeline_running_ref[0] = True
    logger.info(
        f"🚀 PDF PIPELINE STARTED — stem={batch_stem} "
        f"file={Path(pdf_path).name} workers={PDF_BATCH_SIZE}"
    )

    vr: Dict[str, Any] = {
        "user_id":           user_id,
        "input_type":        "document",
        "pdf_path":          pdf_path,
        "video_path":        batch_stem,
        "frames_index":      [],
        "per_frame_details": [],
        "lecture_summary":   {},
        "audio_topics":      {},
        "study_notes":       None,
        "flashcards":        [],
        "quiz":              [],
        "knowledge_graph":   None,
        "pdf_report_path":   None,
        "deduped_concepts":  [],
        "deduped_formulas":  [],
        "error":             None,
        "page_count":        0,
        "backends":          _check_pdf_backends(),
    }
    academic_results[batch_stem] = vr

    try:
        if not FITZ_AVAILABLE and not PDFPLUMBER_AVAILABLE:
            raise RuntimeError(
                "PDF backends not installed. Run: pip install pymupdf pdfplumber"
            )

        n_pages = _page_count(pdf_path)
        if n_pages == 0:
            raise RuntimeError("PDF appears to be empty or unreadable.")
        vr["page_count"] = n_pages
        logger.info(f"[PDF Pipeline] {n_pages} pages detected.")

        # ── Models ────────────────────────────────────────────────────────────
        ocr = OCRExtractor(use_gpu=False)

        llm = shared_llm
        if llm is None:
            logger.info("[PDF Pipeline] No shared LLM — loading fresh Phi-3 instance.")
            llm = LlamaReasoner(
                model_id       = config.reasoning_model_id,
                max_new_tokens = config.max_reasoning_tokens,
                device         = setup_device(),
                load_in_4bit   = config.phi3_load_in_4bit,
                adapter_path   = getattr(config, "phi3_adapter_path", None) or None,
            )

        # ── FIX 4: register back into main._shared_llm RIGHT NOW ─────────────
        # Must happen before any await so the flashcard endpoint can find it
        # even if it is called while the pipeline is still running.
        _register_shared_llm(llm, shared_llm_ref)

        lang_info = _lang_detector.from_code("en") if _lang_detector else {"code": "en"}

        # ── Render pages + extract text concurrently (pure I/O) ───────────────
        loop = asyncio.get_event_loop()
        logger.info(f"[PDF Pipeline] Rendering pages at {PAGE_RENDER_DPI} DPI…")
        t0 = time.time()

        page_dir = os.path.join(IMAGE_DIR, batch_stem)
        image_paths, page_texts = await asyncio.gather(
            loop.run_in_executor(
                None, _render_pages_to_images, pdf_path, page_dir, batch_stem
            ),
            loop.run_in_executor(
                None, _extract_text_per_page, pdf_path
            ),
        )

        logger.info(
            f"[PDF Pipeline] Render + text extraction: {time.time() - t0:.1f}s — "
            f"{len(image_paths)} images, {len(page_texts)} text pages."
        )

        while len(image_paths) < n_pages:
            image_paths.append("")
        while len(page_texts) < n_pages:
            page_texts.append("")

        # ── Phase 1: concurrent page analysis ─────────────────────────────────
        logger.info(
            f"[PDF Pipeline] Phase 1 — analysing {n_pages} pages "
            f"({PDF_BATCH_SIZE} concurrent workers)…"
        )
        t1 = time.time()

        page_analyses = await _analyse_pages_concurrent(
            n_pages, image_paths, page_texts, ocr, llm, lang_info, vr
        )

        logger.info(
            f"[PDF Pipeline] Phase 1 done in {time.time() - t1:.1f}s — "
            f"{len(page_analyses)} pages."
        )

        vr["frames_index"] = page_analyses
        pages_for_llm = (
            page_analyses[:MAX_PAGES_PHASE2]
            if len(page_analyses) > MAX_PAGES_PHASE2
            else page_analyses
        )

        # ── Phase 2, Call 1: Metadata ─────────────────────────────────────────
        logger.info("[PDF Pipeline] Phase 2, Call 1 — extracting metadata…")
        call1_prompt = prompt_image_batch_metadata(
            batch_stem, pages_for_llm, min(5, len(pages_for_llm)), 8
        )
        if _lang_detector:
            call1_prompt = _lang_detector.patch_prompt(call1_prompt, lang_info)

        meta: Dict[str, Any] = {}
        try:
            raw_result = llm.reason(call1_prompt)
            if isinstance(raw_result, dict):
                meta = _serialize(raw_result)
            else:
                raw_text = getattr(llm, "_last_raw_output", "") or str(raw_result)
                meta = _parse_llm_json(raw_text, context=f"{batch_stem}/meta")
        except Exception as exc:
            logger.error(f"[PDF Pipeline] Call 1 failed: {exc}", exc_info=True)
            raw_text = getattr(llm, "_last_raw_output", "") or ""
            meta = _parse_llm_json(raw_text, context=f"{batch_stem}/meta")

        if not isinstance(meta, dict):
            logger.warning(
                f"[PDF/{batch_stem}] metadata resolved to "
                f"{type(meta).__name__} — using empty dict."
            )
            meta = {}

        lecture_title = meta.get("lecture_title") or Path(pdf_path).stem
        subject_area  = meta.get("subject_area", "General")
        topics        = meta.get("topics", [])
        key_concepts  = meta.get("key_concepts", [])
        outcomes      = meta.get("learning_outcomes", [])
        summary       = meta.get("summary", "")
        difficulty    = meta.get("difficulty", "")

        lecture_summary = {
            "lecture_title":     lecture_title,
            "subject_area":      subject_area,
            "main_topics":       topics,
            "learning_outcomes": outcomes,
            "summary":           summary,
            "difficulty_level":  difficulty,
        }
        audio_topics = {
            "lecture_title":    lecture_title,
            "subject_area":     subject_area,
            "topics_covered":   topics,
            "key_concepts":     [{"concept": c, "explanation": ""} for c in key_concepts],
            "important_points": outcomes,
            "summary":          summary,
        }
        logger.info(
            f"[PDF Pipeline] Call 1 done — "
            f"title='{lecture_title[:60]}' subject='{subject_area}'"
        )

        # ── Phase 2, Call 2: Study notes ──────────────────────────────────────
        logger.info("[PDF Pipeline] Phase 2, Call 2 — generating study notes…")
        call2_prompt = prompt_image_study_notes(
            lecture_title, subject_area, difficulty, topics, key_concepts,
            outcomes, summary, pages_for_llm,
        )
        if _lang_detector:
            call2_prompt = _lang_detector.patch_prompt(call2_prompt, lang_info)

        notes_md = ""
        try:
            notes_md = llm.reason_text(call2_prompt)
        except Exception as exc:
            logger.error(f"[PDF Pipeline] Call 2 (notes) failed: {exc}", exc_info=True)

        if not notes_md or not notes_md.strip():
            lines = [f"# Study Notes: {lecture_title}", ""]
            if summary:
                lines += ["## Overview", "", summary, ""]
            if topics:
                lines += ["## Topics", ""] + [f"- {t}" for t in topics] + [""]
            if key_concepts:
                lines += ["## Key Concepts", ""] + [f"- {c}" for c in key_concepts] + [""]
            for pg in page_analyses:
                ac = pg.get("academic_content", {})
                if ac.get("importance") == "low":
                    continue
                lines.append(f"### {pg.get('image_title', pg.get('filename', 'Page'))}")
                if pg.get("description"):
                    lines.append(pg["description"])
                if pg.get("bullet_points"):
                    lines += [f"- {bp}" for bp in pg["bullet_points"]]
                lines.append("")
            notes_md = "\n".join(lines)
            logger.info("[PDF Pipeline] Using fallback notes from metadata.")
        elif not notes_md.lstrip().startswith("#"):
            notes_md = f"# Study Notes: {lecture_title}\n\n" + notes_md

        notes_path = _write_text_file(
            os.path.join(NOTES_DIR, f"{batch_stem}_study_notes.md"),
            notes_md,
        )
        logger.info(f"[PDF Pipeline] Notes → {notes_path}")

        # ── Phase 3: Knowledge graph + PDF report ─────────────────────────────
        logger.info("[PDF Pipeline] Phase 3 — knowledge graph + PDF report…")

        graph_path = None
        graph_d3   = None
        pdf_out    = None

        def _build_graph():
            nonlocal graph_d3, graph_path
            try:
                if _graph_builder:
                    graph      = _graph_builder.build(page_analyses, audio_topics, lecture_summary)
                    graph_d3   = _graph_builder.to_d3_json(graph)
                    graph_path = _graph_builder.save(
                        graph,
                        os.path.join(GRAPH_DIR, f"{batch_stem}_knowledge_graph.json"),
                    )
                    logger.info(f"[PDF Pipeline] Graph → {graph_path}")
            except Exception as exc:
                logger.error(f"[PDF Pipeline] Knowledge graph failed: {exc}", exc_info=True)

        def _build_pdf():
            nonlocal pdf_out
            try:
                pdf_out = generate_pdf_report(
                    batch_stem, PDF_DIR, lecture_summary, audio_topics,
                    page_analyses, [], ""
                )
                logger.info(f"[PDF Pipeline] PDF report → {pdf_out}")
            except Exception as exc:
                logger.error(f"[PDF Pipeline] PDF report failed: {exc}", exc_info=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as out_pool:
            gf = out_pool.submit(_build_graph)
            pf = out_pool.submit(_build_pdf)
            gf.result()
            pf.result()

        # ── Finalise ──────────────────────────────────────────────────────────
        vr.update({
            "lecture_summary":       lecture_summary,
            "audio_topics":          audio_topics,
            "study_notes":           notes_md,
            "study_notes_path":      notes_path,
            "knowledge_graph":       graph_d3,
            "knowledge_graph_path":  graph_path,
            "pdf_report_path":       pdf_out,
            "total_frames_analysed": len(page_analyses),
        })

        flashcard_states[batch_stem] = {
            "state":           "idle",
            "flashcard_count": 0,
            "quiz_count":      0,
            "error":           None,
        }

        if user_id:
            try:
                from db_actions import save_pipeline_result_to_db
                save_pipeline_result_to_db(user_id, pdf_path, vr, batch_stem)
                logger.info(f"[PDF Pipeline] Persisted '{batch_stem}' to DB.")
            except Exception as exc:
                logger.warning(f"[PDF Pipeline] DB persist failed (non-fatal): {exc}")

        logger.info(
            f"✅ PDF PIPELINE COMPLETE — {batch_stem}\n"
            f"   {len(page_analyses)} pages analysed\n"
            f"   Notes:  {notes_path}\n"
            f"   PDF:    {pdf_out}\n"
            f"   Graph:  {graph_path}\n"
            f"   Flashcards: POST /generate/flashcards/{batch_stem}"
        )

    except Exception as exc:
        logger.error(
            f"❌ PDF PIPELINE FAILED — {batch_stem}: {exc}", exc_info=True
        )
        vr["error"] = str(exc)

    finally:
        pipeline_running_ref[0] = False