"""
video_pipeline/core/stream_manager.py
=======================================
Manages one or more video streams and yields batches of frames at a
configurable target FPS.

GPU acceleration
----------------
When opencv-contrib-python is installed and a CUDA-capable GPU is present,
cv2.cudacodec.VideoReader decodes H.264/HEVC frames using the GPU NVDEC
hardware decoder — 2-4x faster than CPU VideoCapture on RTX 3050.

Fallback chain:
  1. cv2.cudacodec.VideoReader  (GPU NVDEC, opencv-contrib + CUDA required)
  2. cv2.VideoCapture           (CPU, always available)

Prefetch thread
---------------
Each VideoStream runs a background thread that reads frames ahead into a
queue (default depth 4). The main async loop reads from the queue —
zero blocking I/O on the hot path.

Install GPU decode:
    pip install opencv-contrib-python
"""
from __future__ import annotations

import asyncio
import queue
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from video_pipeline.utils.logger import get_logger

logger = get_logger(__name__)

_DONE = object()   # sentinel: placed in queue when stream is exhausted

# ── GPU availability ──────────────────────────────────────────────────────────
_CUDA_CODEC_AVAILABLE = False
try:
    _ = cv2.cudacodec
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        _CUDA_CODEC_AVAILABLE = True
        logger.info("[StreamManager] GPU video decoding enabled (cv2.cudacodec).")
    else:
        logger.info("[StreamManager] cv2.cudacodec present but no CUDA device.")
except AttributeError:
    logger.info(
        "[StreamManager] CPU video decoding (cv2.VideoCapture). "
        "For GPU: pip install opencv-contrib-python"
    )


class VideoStream:
    """
    Wraps a single video file with background prefetch and optional GPU decode.
    """

    def __init__(self, stream_id: str, video_path: str, target_fps: float, prefetch: int = 4) -> None:
        self.stream_id  = stream_id
        self.video_path = video_path
        self.target_fps = max(0.1, target_fps)

        # Probe metadata via VideoCapture (cudacodec doesn't expose CAP_PROP_*)
        _probe = cv2.VideoCapture(video_path)
        if not _probe.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        self._native_fps  = _probe.get(cv2.CAP_PROP_FPS) or 25.0
        self._frame_count = int(_probe.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration    = self._frame_count / self._native_fps
        _probe.release()

        self._skip = max(1, int(round(self._native_fps / self.target_fps)))
        self._done = False

        logger.info(
            f"[{stream_id}] Opened {video_path} | "
            f"native_fps={self._native_fps:.1f} frames={self._frame_count} "
            f"duration={self._duration:.1f}s sample_every={self._skip} "
            f"(target_fps={self.target_fps}) gpu={_CUDA_CODEC_AVAILABLE}"
        )

        self._q: queue.Queue = queue.Queue(maxsize=prefetch)
        self._thread = threading.Thread(
            target=self._prefetch_loop, daemon=True, name=f"prefetch_{stream_id}"
        )
        self._thread.start()

    # ── Prefetch thread ───────────────────────────────────────────────────────

    def _prefetch_loop(self) -> None:
        try:
            if _CUDA_CODEC_AVAILABLE:
                self._prefetch_gpu()
            else:
                self._prefetch_cpu()
        except Exception as exc:
            logger.error(f"[{self.stream_id}] Prefetch error: {exc}", exc_info=True)
        finally:
            self._q.put(_DONE)

    def _prefetch_gpu(self) -> None:
        """GPU NVDEC sequential decode — skips non-sampled frames."""
        try:
            reader = cv2.cudacodec.createVideoReader(self.video_path)
        except Exception as exc:
            logger.warning(f"[{self.stream_id}] cudacodec failed ({exc}), using CPU.")
            self._prefetch_cpu()
            return

        frame_idx = 0
        while True:
            ret, gpu_frame = reader.nextFrame()
            if not ret or gpu_frame is None:
                break
            if frame_idx % self._skip == 0:
                # Download GPU→CPU (OCR + slide detector need numpy)
                self._q.put((gpu_frame.download(), frame_idx / self._native_fps))
            frame_idx += 1
        logger.info(f"[{self.stream_id}] GPU decode exhausted at frame {frame_idx}.")

    def _prefetch_cpu(self) -> None:
        """CPU decode with direct seek to sampled frame indices.

        Fix: after seek, verify the actual position before accepting the frame.
        CAP_PROP_POS_FRAMES seek is not guaranteed to land exactly on the
        requested frame for all codecs — a mismatch means the timestamp
        stored alongside the frame would be wrong, silently corrupting the
        slide timeline fed to the LLM.
        """
        cap   = cv2.VideoCapture(self.video_path)
        index = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            # Read actual position AFTER seek to use the real timestamp
            actual_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            # Use the real decoded position for the timestamp, not the
            # requested index, so timestamps are accurate even when the
            # codec snaps to the nearest keyframe.
            real_ts = actual_pos / self._native_fps
            self._q.put((frame, real_ts))
            index += self._skip
        cap.release()
        logger.info(f"[{self.stream_id}] CPU decode exhausted at frame {index}.")

    # ── Public API ────────────────────────────────────────────────────────────

    def read_next(self) -> Optional[Tuple[np.ndarray, float]]:
        if self._done:
            return None
        item = self._q.get()
        if item is _DONE:
            self._done = True
            return None
        return item

    @property
    def is_done(self) -> bool:
        return self._done

    def release(self) -> None:
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        logger.debug(f"[{self.stream_id}] Released.")


class StreamManager:
    """Manages multiple VideoStreams, returns one frame per stream per call."""

    def __init__(self, sources: Dict[str, str], target_fps: float = 1.0, prefetch: int = 4) -> None:
        self._streams: Dict[str, VideoStream] = {}
        for sid, path in sources.items():
            try:
                self._streams[sid] = VideoStream(sid, path, target_fps, prefetch)
            except RuntimeError as exc:
                logger.error(f"Failed to open stream {sid}: {exc}")
        logger.info(
            f"StreamManager: {len(self._streams)} stream(s) at {target_fps} FPS "
            f"gpu_decode={_CUDA_CODEC_AVAILABLE}."
        )

    async def get_batch(self) -> Tuple[List[str], List[np.ndarray], List[float]]:
        """Read one prefetched frame per active stream. Near-instant — no disk I/O."""
        await asyncio.sleep(0)

        stream_ids: List[str]        = []
        frames:     List[np.ndarray] = []
        timestamps: List[float]      = []
        exhausted:  List[str]        = []

        for sid, stream in self._streams.items():
            result = stream.read_next()
            if result is None:
                exhausted.append(sid)
            else:
                stream_ids.append(sid)
                frames.append(result[0])
                timestamps.append(result[1])

        for sid in exhausted:
            self._streams[sid].release()
            del self._streams[sid]
            logger.info(f"Stream {sid} removed (exhausted).")

        if not self._streams and not frames:
            return [], [], []
        return stream_ids, frames, timestamps

    def release_all(self) -> None:
        for stream in self._streams.values():
            stream.release()
        self._streams.clear()
        logger.info("StreamManager: all streams released.")