"""
video_pipeline/detection/ocr.py
=================================
OCR text extractor for lecture frames and slide images.

Primary backend  : EasyOCR   (GPU-capable)
Fallback backend : Tesseract (via pytesseract)

v3: set_languages() for runtime language switching.
v4: GPU preprocessing via cv2.cuda (BGR→Gray, resize ~3x faster on RTX 3050)
    + batch_extract() to process multiple frames in one EasyOCR call.

v4.0.1 fixes (matches config.py v3.0.3)
-----------------------------------------
  use_gpu default changed False → True.
    OCRExtractor was constructed with use_gpu=False everywhere, meaning GPU
    was NEVER used even when the RTX 3050 had ~1.6 GB spare VRAM after
    Phi-3 4-bit loaded. This caused all 40 frames to return empty OCR on
    animated lecture videos, triggering the transcript-only fallback and
    ultimately producing no notes / flashcards / quiz / PDF.

  config.easyocr_gpu now controls the flag at construction time.
    Import config at the top of this file and pass config.easyocr_gpu as
    the default for use_gpu so a single env-var (EASYOCR_GPU=false) can
    force CPU mode if OOM errors appear after enabling GPU OCR.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from video_pipeline.config import config  # v4.0.1: wire GPU flag to config

logger = logging.getLogger(__name__)

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    logger.info("[OCR] EasyOCR available [OK]")
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("[OCR] EasyOCR not installed. Will try Tesseract.")

try:
    import pytesseract
    from PIL import Image as PILImage
    TESSERACT_AVAILABLE = True
    logger.info("[OCR] Tesseract (pytesseract) available [OK]")
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("[OCR] pytesseract not installed.")

_CV2_CUDA_AVAILABLE = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        _CV2_CUDA_AVAILABLE = True
        logger.info("[OCR] cv2.cuda available — GPU image preprocessing enabled.")
except Exception:
    pass


def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """BGR → grayscale + upscale to min 480px height. GPU-accelerated when available."""
    if _CV2_CUDA_AVAILABLE:
        try:
            g = cv2.cuda_GpuMat()
            g.upload(frame)
            gray = cv2.cuda.cvtColor(g, cv2.COLOR_BGR2GRAY).download()
        except Exception:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if h < 480:
        scale = 480 / h
        new_w = int(w * scale)
        if _CV2_CUDA_AVAILABLE:
            try:
                g = cv2.cuda_GpuMat()
                g.upload(gray)
                gray = cv2.cuda.resize(g, (new_w, 480)).download()
            except Exception:
                gray = cv2.resize(gray, (new_w, 480), interpolation=cv2.INTER_CUBIC)
        else:
            gray = cv2.resize(gray, (new_w, 480), interpolation=cv2.INTER_CUBIC)
    return gray


class OCRExtractor:
    """
    Extracts text from BGR frames using EasyOCR (preferred) or Tesseract.

    v4 additions:
      - _preprocess_frame() uses cv2.cuda for BGR→Gray + resize when available
      - batch_extract(frames) processes N frames in one EasyOCR call (~30% faster)

    v4.0.1 fix:
      - use_gpu now defaults to config.easyocr_gpu (True on RTX 3050) instead
        of hardcoded False. Pass use_gpu=False explicitly only for testing.
    """

    _reader_cache: Dict[str, Any] = {}

    def __init__(
        self,
        languages: List[str] = None,
        # v4.0.1 FIX: was hardcoded False — GPU was never used even when VRAM
        # was available. Now reads from config so EASYOCR_GPU=false can override.
        use_gpu: bool = config.easyocr_gpu,
    ) -> None:
        self.use_gpu   = use_gpu
        self.languages = languages or ["en"]
        self._backend  = "none"
        self._reader: Optional[Any] = None
        self._hf_client: Optional[Any] = None

        # OCR_BACKEND=api  -> extract text via a vision model on the HF
        #                     Inference API instead of loading EasyOCR
        #                     locally. Use this on CPU-only / low-RAM hosts
        #                     (e.g. Render free/starter) to avoid the memory
        #                     hit of holding EasyOCR + Whisper + the
        #                     sentence-transformer model in one process.
        # OCR_BACKEND=local (default) -> existing EasyOCR/Tesseract behaviour.
        if os.environ.get("OCR_BACKEND", "local").lower() == "api":
            self._init_api_backend()
            return

        logger.info(f"[OCR] Loading EasyOCR reader (langs={self.languages}, gpu={self.use_gpu}) …")

        if EASYOCR_AVAILABLE:
            self._load_easyocr(self.languages)
        elif TESSERACT_AVAILABLE:
            self._backend = "tesseract"
            logger.info("[OCR] Using Tesseract backend.")
        else:
            logger.error("[OCR] No OCR backend available.")

    def _init_api_backend(self) -> None:
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            logger.error(
                "[OCR/API] huggingface_hub not installed. Run: pip install huggingface_hub"
            )
            self._backend = "none"
            return

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.error(
                "[OCR/API] HF_TOKEN environment variable not set — cannot use OCR_BACKEND=api."
            )
            self._backend = "none"
            return

        self._hf_model_id = os.environ.get("OCR_HF_MODEL_ID", "meta-llama/Llama-3.2-11B-Vision-Instruct")
        self._hf_client = InferenceClient(model=self._hf_model_id, token=hf_token)
        self._backend = "api"
        logger.info(f"[OCR/API] Ready — using HF Inference API model '{self._hf_model_id}'.")

    def set_languages(self, languages: List[str]) -> None:
        if not EASYOCR_AVAILABLE:
            return
        if sorted(languages) == sorted(self.languages):
            return
        logger.info(f"[OCR] Switching languages: {self.languages} → {languages}")
        self.languages = languages
        self._load_easyocr(languages)

    def extract(self, frame: np.ndarray, conf_threshold: float = 0.4) -> List[Dict[str, Any]]:
        """Run OCR on a single BGR frame."""
        if self._backend == "none":
            return []
        if self._backend == "api":
            # Send the original color frame — vision models generally do
            # better OCR on color images than the grayscale-preprocessed
            # version used for EasyOCR/Tesseract.
            return self._api_ocr(frame)
        gray = _preprocess_frame(frame)
        try:
            if self._backend == "easyocr":
                return self._easyocr(gray, conf_threshold)
            return self._tesseract(gray, conf_threshold)
        except Exception as exc:
            logger.warning(f"[OCR] Extraction failed: {exc}")
            return []

    def batch_extract(
        self,
        frames: List[np.ndarray],
        conf_threshold: float = 0.4,
    ) -> List[List[Dict[str, Any]]]:
        """
        Process multiple frames in a single EasyOCR call.

        EasyOCR's readtext_batched() shares GPU upload overhead across all
        frames — ~30% faster than N separate extract() calls when N >= 4.
        Falls back to sequential extract() for Tesseract or older EasyOCR.
        """
        if self._backend == "none" or not frames:
            return [[] for _ in frames]
        if self._backend != "easyocr":
            return [self.extract(f, conf_threshold) for f in frames]

        grays = [_preprocess_frame(f) for f in frames]
        try:
            all_results = self._reader.readtext_batched(grays, detail=1, paragraph=False)
            output = []
            for frame_results in all_results:
                parsed = []
                for bbox, text, conf in frame_results:
                    if conf >= conf_threshold and text.strip():
                        parsed.append({
                            "text":       text.strip(),
                            "confidence": round(float(conf), 4),
                            "bbox":       bbox,
                        })
                output.append(parsed)
            return output
        except Exception as exc:
            logger.debug(f"[OCR] batch fallback to sequential ({exc})")
            return [self.extract(f, conf_threshold) for f in frames]

    def results_to_text(self, results: List[Dict[str, Any]]) -> str:
        return " ".join(r["text"] for r in results if r.get("text")).strip()

    @property
    def backend(self) -> str:
        return self._backend

    def _load_easyocr(self, languages: List[str]) -> None:
        # Cache key includes gpu flag — GPU and CPU readers are separate objects
        key = ",".join(sorted(languages)) + f":gpu={self.use_gpu}"
        if key not in OCRExtractor._reader_cache:
            logger.info(f"[OCR] Loading EasyOCR reader (langs={languages}, gpu={self.use_gpu}) …")
            try:
                OCRExtractor._reader_cache[key] = easyocr.Reader(
                    languages, gpu=self.use_gpu, verbose=False,
                )
                logger.info("[OCR] EasyOCR reader loaded [OK]")
            except Exception as exc:
                logger.error(f"[OCR] EasyOCR reader init failed: {exc}")
                if TESSERACT_AVAILABLE:
                    self._backend = "tesseract"
                return
        self._reader  = OCRExtractor._reader_cache[key]
        self._backend = "easyocr"

    def _easyocr(self, gray: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
        results = []
        for bbox, text, conf in self._reader.readtext(gray, detail=1, paragraph=False):
            if conf >= threshold and text.strip():
                results.append({"text": text.strip(), "confidence": round(float(conf), 4), "bbox": bbox})
        return results

    def _tesseract(self, gray: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
        pil_img = PILImage.fromarray(gray)
        data    = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, config="--psm 6")
        results = []
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i]) / 100.0
            except (ValueError, TypeError):
                conf = 0.0
            if conf >= threshold:
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                results.append({"text": text, "confidence": round(conf, 4),
                                 "bbox": [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]})
        return results

    def _api_ocr(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract text from a frame via a vision-capable model on the HF
        Inference API. Returns the same [{"text", "confidence", "bbox"}]
        shape as the local backends so ocr_to_text()/results_to_text()
        downstream need no changes — bbox is always None here since a
        chat-style vision model returns free text, not per-region boxes
        (nothing downstream in this pipeline actually uses bbox — it's
        flattened to plain text immediately after extraction).
        """
        if self._hf_client is None:
            return []
        try:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                logger.warning("[OCR/API] Failed to encode frame as JPEG.")
                return []
            b64_image = base64.b64encode(buf.tobytes()).decode("utf-8")
            data_uri = f"data:image/jpeg;base64,{b64_image}"

            response = self._hf_client.chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": (
                            "Transcribe all visible text in this image exactly as written. "
                            "Only output the transcribed text, nothing else. "
                            "If there is no readable text, output nothing."
                        )},
                    ],
                }],
                max_tokens=512,
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                return []
            return [{"text": text, "confidence": 1.0, "bbox": None}]

        except Exception as exc:
            logger.warning(f"[OCR/API] Extraction failed: {exc}")
            return []