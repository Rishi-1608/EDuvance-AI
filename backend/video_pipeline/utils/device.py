"""
video_pipeline/utils/device.py
================================
Detects the best available compute device (CUDA GPU → MPS → CPU).

Usage:
    from video_pipeline.utils.device import setup_device
    device = setup_device()   # returns "cuda", "mps", or "cpu"
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def setup_device() -> str:
    """
    Return the best available PyTorch device string.

    Priority: CUDA GPU → Apple MPS → CPU
    Falls back gracefully if torch is not installed.
    """
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            name   = torch.cuda.get_device_name(0)
            vram   = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"Device: CUDA — {name} ({vram:.1f} GB VRAM)")
            if vram < 5.0:
                logger.info(
                    "  VRAM < 5 GB detected — 4-bit quantisation (PHI3_LOAD_IN_4BIT=true) "
                    "is required and is enabled by default."
                )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Device: Apple MPS (Metal Performance Shaders)")
        else:
            device = "cpu"
            logger.info("Device: CPU")

        return device

    except ImportError:
        logger.warning("torch not installed — defaulting to CPU.")
        return "cpu"