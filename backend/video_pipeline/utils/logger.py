"""
video_pipeline/utils/logger.py
================================
Centralised logging factory.

Usage:
    from video_pipeline.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started")
"""
from __future__ import annotations

import logging
import os
import sys

LOG_LEVEL   = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Configure root logger once
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stdout,
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger inheriting the root configuration."""
    return logging.getLogger(name)