"""
metadata/video_metadata.py
============================
Extracts technical metadata from a video file using OpenCV (cv2).
Falls back to ffprobe (subprocess) for extended format information.

Returns a structured dict:
{
  "summary": {
    "filename":     "lecture.mp4",
    "duration_hms": "00:45:12",
    "duration_sec": 2712.0,
    "resolution":   "1920x1080",
    "fps":          30.0,
    "total_frames": 81360,
    "codec":        "mp4v",
    "file_size_mb": 542.3,
  },
  "ffprobe": { ... }   # extended info if ffprobe available
}

Usage:
    from metadata.video_metadata import extract_video_metadata
    meta = extract_video_metadata("uploads/lecture.mp4")
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

logger = logging.getLogger(__name__)


def _seconds_to_hms(seconds: float) -> str:
    """Convert a float number of seconds to 'HH:MM:SS' string."""
    s   = int(seconds)
    hh  = s // 3600
    mm  = (s % 3600) // 60
    ss  = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _run_ffprobe(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Run ffprobe to get extended stream/format information.
    Returns None if ffprobe is not installed or fails.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if result.returncode == 0:
            return json.loads(result.stdout.decode("utf-8", errors="replace"))
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return None


def extract_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a video file.

    Args:
        video_path: Absolute or relative path to the video file.

    Returns:
        Dict with "summary" (always present) and "ffprobe" (if available).
    """
    path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))

    fps          = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int   = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec        = "".join([
        chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)
    ]).strip()

    cap.release()

    duration_sec = (total_frames / fps) if fps > 0 else 0.0
    file_size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0

    summary: Dict[str, Any] = {
        "filename":     path.name,
        "duration_hms": _seconds_to_hms(duration_sec),
        "duration_sec": round(duration_sec, 2),
        "resolution":   f"{width}x{height}",
        "width_px":     width,
        "height_px":    height,
        "fps":          round(fps, 3),
        "total_frames": total_frames,
        "codec":        codec,
        "file_size_mb": round(file_size_mb, 2),
    }

    logger.info(
        f"[Metadata] {path.name} — "
        f"{summary['duration_hms']} @ {summary['resolution']} "
        f"{fps:.1f}fps  {file_size_mb:.1f}MB"
    )

    ffprobe_data = _run_ffprobe(str(video_path))

    # Extract audio stream info from ffprobe
    audio_info: Optional[Dict] = None
    if ffprobe_data:
        for stream in ffprobe_data.get("streams", []):
            if stream.get("codec_type") == "audio":
                audio_info = {
                    "codec":       stream.get("codec_name"),
                    "sample_rate": stream.get("sample_rate"),
                    "channels":    stream.get("channels"),
                    "duration":    stream.get("duration"),
                }
                break

    summary["has_audio"] = audio_info is not None
    summary["audio"]     = audio_info

    return {
        "summary":  summary,
        "ffprobe":  ffprobe_data,
    }