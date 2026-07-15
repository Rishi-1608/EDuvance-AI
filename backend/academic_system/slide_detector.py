"""
academic_system/slide_detector.py
===================================
Slide change detection using frame differencing.

Two signals combined:
  1. Histogram correlation  — fast CPU, catches colour/brightness shifts
  2. SSIM (structural)      — catches text/layout changes on similar slides

v4: Frame resize uses cv2.cuda when available (~3x faster on RTX 3050).
    SSIM stays on CPU — it's already fast on the small 320x240 thumbnail.
"""
from __future__ import annotations

from typing import Optional
import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

_CV2_CUDA_AVAILABLE = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        _CV2_CUDA_AVAILABLE = True
except Exception:
    pass


class SlideChangeDetector:
    """
    Stateful detector — compares each frame to the last accepted keyframe.

    Parameters
    ----------
    hist_threshold : float
        Histogram correlation below which a slide change is declared (0–1).
    ssim_threshold : float
        SSIM score below which a structural change is declared (0–1).
    min_seconds_between_slides : float
        Minimum gap between accepted slide changes (prevents flicker).
    resize_to : tuple[int,int]
        Resize before comparison for speed. Smaller = faster but less precise.
    """

    def __init__(
        self,
        hist_threshold:             float = 0.92,
        ssim_threshold:             float = 0.85,
        min_seconds_between_slides: float = 1.0,
        resize_to:                  tuple = (320, 240),
    ) -> None:
        self.hist_threshold             = hist_threshold
        self.ssim_threshold             = ssim_threshold
        self.min_seconds_between_slides = min_seconds_between_slides
        self.resize_to                  = resize_to

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_hist: Optional[np.ndarray] = None
        self._last_ts:   float                = -999.0

    def is_new_slide(self, frame: np.ndarray, timestamp: float = 0.0) -> bool:
        """Return True if frame represents a new slide (or is the first frame)."""
        small = self._resize(frame)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hist  = self._hist(small)

        if self._prev_hist is None:
            self._accept(gray, hist, timestamp)
            return True

        if (timestamp - self._last_ts) < self.min_seconds_between_slides:
            return False

        if self._changed(gray, hist):
            self._accept(gray, hist, timestamp)
            return True
        return False

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_hist = None
        self._last_ts   = -999.0

    # ── internals ─────────────────────────────────────────────────────────────

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if self.resize_to is None:
            return frame
        # GPU resize when available — this is called for every frame so the
        # speedup accumulates significantly over 100+ frames per video.
        if _CV2_CUDA_AVAILABLE:
            try:
                g = cv2.cuda_GpuMat()
                g.upload(frame)
                return cv2.cuda.resize(g, self.resize_to).download()
            except Exception:
                pass
        return cv2.resize(frame, self.resize_to, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _hist(bgr: np.ndarray) -> np.ndarray:
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def _changed(self, gray: np.ndarray, hist: np.ndarray) -> bool:
        corr = cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL)
        if corr < self.hist_threshold:
            return True
        if SKIMAGE_AVAILABLE and self._prev_gray is not None:
            score = _ssim(self._prev_gray, gray, data_range=255)
            if score < self.ssim_threshold:
                return True
        return False

    def _accept(self, gray: np.ndarray, hist: np.ndarray, ts: float) -> None:
        self._prev_gray = gray
        self._prev_hist = hist
        self._last_ts   = ts