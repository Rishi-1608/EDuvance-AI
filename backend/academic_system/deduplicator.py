"""
academic_system/deduplicator.py
================================
Semantic deduplication of extracted concepts, definitions, and formulas.

Lecture slides often repeat the same idea with slightly different wording.
Without deduplication study notes end up with 10 near-identical lines for
"F = ma" or "Newton's second law".

Two backends, in priority order:
  A. sentence-transformers  (all-MiniLM-L6-v2, ~80 MB) — best quality
  B. TF-IDF + cosine        (sklearn)  — zero extra download, still good
  C. exact-string fallback  — no deps at all

Usage
-----
    dedup = SemanticDeduplicator()

    concepts = dedup.deduplicate(
        ["Newton's second law", "F = ma",
         "force equals mass times acceleration", "kinetic energy"],
        threshold=0.85,
    )
    # → ["Newton's second law", "kinetic energy"]
      (or keeps the longer variant, depending on prefer_longer)
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer as _ST
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer as _TFIDF
    from sklearn.metrics.pairwise import cosine_similarity as _cosine
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SemanticDeduplicator:
    """
    Deduplicates lists of strings by semantic similarity.

    Parameters
    ----------
    model_name : str
        sentence-transformers model ID (ignored if ST not available).
    threshold : float
        Cosine similarity above which two strings are treated as duplicates.
    prefer_longer : bool
        Keep the longer string when merging two duplicates.
    """

    def __init__(
        self,
        model_name:    str   = _DEFAULT_MODEL,
        threshold:     float = 0.85,
        prefer_longer: bool  = True,
    ) -> None:
        self.threshold     = threshold
        self.prefer_longer = prefer_longer
        self._model        = None
        self._backend      = "exact"

        if ST_AVAILABLE:
            try:
                self._model   = _ST(model_name)
                self._backend = "sentence_transformers"
            except Exception:
                pass

        if self._backend == "exact" and SKLEARN_AVAILABLE:
            self._backend = "tfidf"

    # ── public API ────────────────────────────────────────────────────────────

    def deduplicate(
        self,
        items:     List[str],
        threshold: Optional[float] = None,
    ) -> List[str]:
        """Return a deduplicated list, keeping the most descriptive item per cluster."""
        items = [s.strip() for s in items if s and s.strip()]
        if len(items) <= 1:
            return items

        t = threshold if threshold is not None else self.threshold

        # Normalise to lowercase for embedding/comparison only.
        # We keep a parallel list of originals so we return the best-cased version.
        normalised = [s.lower() for s in items]
        embs = self._embed(normalised)
        sim  = self._sim_matrix(embs)

        kept:    List[int] = []
        dropped: set       = set()

        for i in range(len(items)):
            if i in dropped:
                continue
            kept.append(i)
            for j in range(i + 1, len(items)):
                if j in dropped:
                    continue
                if sim[i, j] >= t:
                    # Replace kept index if j is longer (more descriptive)
                    if self.prefer_longer and len(items[j]) > len(items[kept[-1]]):
                        kept[-1] = j
                    dropped.add(j)

        return [items[i] for i in kept]

    def deduplicate_concepts(self, frames: List[Dict], threshold: float = 0.85) -> List[str]:
        """
        Extract + deduplicate key_concepts across all frame dicts.

        Fix: threshold was hardcoded to 0.82 here while the constructor default
        is 0.85, causing more aggressive merging of concepts than intended.
        Now defaults to 0.85 to match the constructor.

        Handles two formats that appear in the pipeline:
          - plain strings:  ["Newton's Second Law", "force"]
          - dicts:          [{"concept": "Newton's Second Law", "explanation": "..."}]
            (produced by prompt_audio_topics)
        """
        raw: List[str] = []
        for f in frames:
            ac = f.get("academic_content", {})
            if isinstance(ac, dict):
                for item in ac.get("key_concepts", []):
                    if isinstance(item, dict):
                        val = item.get("concept") or item.get("name") or item.get("term") or ""
                    else:
                        val = str(item)
                    if val.strip():
                        raw.append(val.strip())
        return self.deduplicate(raw, threshold=threshold)

    def deduplicate_definitions(self, frames: List[Dict], threshold: float = 0.88) -> List[Dict]:
        """
        Deduplicate definition dicts by term similarity.
        Keeps the longer definition text when merging.
        """
        all_defs: List[Dict] = []
        for f in frames:
            ac = f.get("academic_content", {})
            if isinstance(ac, dict):
                all_defs.extend(ac.get("definitions", []))
        if not all_defs:
            return []

        terms       = [d.get("term", "") for d in all_defs]
        dedup_terms = self.deduplicate(terms, threshold=threshold)

        # Build map: canonical term → richest definition dict
        result: Dict[str, Dict] = {}
        for term in dedup_terms:
            for d in all_defs:
                if d.get("term", "") == term:
                    existing = result.get(term)
                    if not existing or len(d.get("definition", "")) > len(existing.get("definition", "")):
                        result[term] = d
                    break
        return list(result.values())

    def deduplicate_formulas(self, frames: List[Dict], threshold: float = 0.92) -> List[str]:
        """Extract + deduplicate formulas. Higher threshold — formulas are short."""
        raw: List[str] = []
        for f in frames:
            ac = f.get("academic_content", {})
            if isinstance(ac, dict):
                raw.extend(ac.get("formulas", []))
        normalised = [re.sub(r"\s+", " ", s).strip() for s in raw]
        return self.deduplicate(normalised, threshold=threshold)

    @property
    def backend(self) -> str:
        return self._backend

    # ── embedding backends ────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        if self._backend == "sentence_transformers" and self._model is not None:
            return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if self._backend == "tfidf":
            return self._tfidf_embed(texts)
        # Exact fallback: identity matrix (no merging unless identical)
        return np.eye(len(texts))

    @staticmethod
    def _tfidf_embed(texts: List[str]) -> np.ndarray:
        try:
            vec = _TFIDF(ngram_range=(1, 2), min_df=1)
            return vec.fit_transform(texts).toarray()
        except Exception:
            return np.eye(len(texts))

    @staticmethod
    def _sim_matrix(embeddings: np.ndarray) -> np.ndarray:
        if SKLEARN_AVAILABLE:
            return _cosine(embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        n = embeddings / norms
        return n @ n.T