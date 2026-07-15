"""
minio_utils.py  —  AcademIQ MinIO client  (updated for database_v3)
=====================================================================

Changes vs original
--------------------
① get_presigned_url accepts a timedelta for expires_in (not raw seconds)
  and returns a cached URL so repeated calls for the same object within
  the TTL window don't hit MinIO on every request.

② delete_object / delete_prefix added (used by DELETE /api/v1/lecture/{stem}).

③ object_exists() — cheap HEAD check before attempting a download.

④ get_presigned_url_for_asset() accepts a MediaAsset ORM object directly,
  reading (minio_bucket, minio_key) from it — the canonical pattern.

⑤ All public methods log at INFO on success and ERROR on failure,
  consistent with the rest of the codebase.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import timedelta
from typing import Optional

from minio import Minio
from minio.error import S3Error

from video_pipeline.utils.logger import get_logger

logger = get_logger(__name__)

# ── config ─────────────────────────────────────────────────────────────────────
# Works against real MinIO OR any S3-compatible provider (e.g. Cloudflare R2)
# by just changing these env vars — no code changes needed.
#
# Cloudflare R2 example:
#   MINIO_ENDPOINT   = <account_id>.r2.cloudflarestorage.com
#   MINIO_ACCESS_KEY = <R2 access key id>
#   MINIO_SECRET_KEY = <R2 secret access key>
#   MINIO_SECURE     = true
#   MINIO_REGION     = auto
MINIO_ENDPOINT    = os.environ.get("MINIO_ENDPOINT",    "localhost:9000")
MINIO_ACCESS_KEY  = os.environ.get("MINIO_ACCESS_KEY",  "minioadmin")
MINIO_SECRET_KEY  = os.environ.get("MINIO_SECRET_KEY",  "minioadmin")
MINIO_SECURE      = os.environ.get("MINIO_SECURE",      "false").lower() == "true"
MINIO_BUCKET_NAME = os.environ.get("MINIO_BUCKET_NAME", "media")
MINIO_REGION      = os.environ.get("MINIO_REGION",      None)  # R2 needs "auto"

# Presigned URL TTL — objects are served for this long before a new URL is issued
_PRESIGN_TTL_SEC  = int(os.environ.get("MINIO_PRESIGN_TTL_SEC", str(3600)))   # 1 h default

# ── simple in-process URL cache (thread-safe) ──────────────────────────────────
# Structure: { (bucket, key) → (url: str, expires_at: float) }
_url_cache: dict = {}
_cache_lock = threading.Lock()


class MinioClient:
    def __init__(self) -> None:
        self.client = Minio(
            MINIO_ENDPOINT,
            access_key = MINIO_ACCESS_KEY,
            secret_key = MINIO_SECRET_KEY,
            secure     = MINIO_SECURE,
            region     = MINIO_REGION,
        )
        self.MINIO_BUCKET_NAME = MINIO_BUCKET_NAME
        self._ensure_bucket_exists(MINIO_BUCKET_NAME)

    # ── internal ───────────────────────────────────────────────────────────────

    def _ensure_bucket_exists(self, bucket: str) -> None:
        try:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
                logger.info(f"[MinIO] Created bucket: {bucket}")
            else:
                logger.info(f"[MinIO] Bucket exists: {bucket}")
        except S3Error as exc:
            logger.error(f"[MinIO] bucket check/create error: {exc}")
            raise

    # ── upload ─────────────────────────────────────────────────────────────────

    def upload_file(self, file_path: str, object_name: str, bucket: Optional[str] = None) -> str:
        """
        Upload a local file to MinIO.
        Returns the full MinIO path: '<bucket>/<object_name>'.
        """
        bucket = bucket or self.MINIO_BUCKET_NAME
        try:
            self.client.fput_object(bucket, object_name, file_path)
            logger.info(f"[MinIO] Uploaded {file_path!r} → {bucket}/{object_name}")
            return f"{bucket}/{object_name}"
        except S3Error as exc:
            logger.error(f"[MinIO] Upload failed for {file_path!r}: {exc}")
            raise

    def upload_stream(
        self,
        data,
        object_name:  str,
        length:       int,
        content_type: str = "application/octet-stream",
        bucket:       Optional[str] = None,
    ) -> str:
        bucket = bucket or self.MINIO_BUCKET_NAME
        try:
            self.client.put_object(bucket, object_name, data, length, content_type=content_type)
            logger.info(f"[MinIO] Stream uploaded → {bucket}/{object_name}")
            return f"{bucket}/{object_name}"
        except S3Error as exc:
            logger.error(f"[MinIO] Stream upload failed for {object_name!r}: {exc}")
            raise

    # ── presigned URLs (cached) ────────────────────────────────────────────────

    def get_presigned_url(
        self,
        object_name:       str,
        bucket:            Optional[str] = None,
        expires_in_seconds: int          = _PRESIGN_TTL_SEC,
    ) -> str:
        """
        Return a presigned GET URL for the object.
        Results are cached for (expires_in_seconds - 60) seconds so the
        URL is still valid when the client actually uses it.
        """
        bucket = bucket or self.MINIO_BUCKET_NAME
        cache_key = (bucket, object_name)

        with _cache_lock:
            cached = _url_cache.get(cache_key)
            if cached:
                url, expires_at = cached
                if time.monotonic() < expires_at:
                    return url

        try:
            url = self.client.presigned_get_object(
                bucket,
                object_name,
                expires=timedelta(seconds=expires_in_seconds),
            )
            # cache with a 60-second safety margin
            with _cache_lock:
                _url_cache[cache_key] = (url, time.monotonic() + expires_in_seconds - 60)
            return url
        except S3Error as exc:
            logger.error(f"[MinIO] Presign failed for {bucket}/{object_name}: {exc}")
            raise

    def get_presigned_url_for_asset(self, asset, expires_in_seconds: int = _PRESIGN_TTL_SEC) -> str:
        """
        Convenience: accepts a MediaAsset ORM instance directly.
        Usage: url = minio_client.get_presigned_url_for_asset(asset)
        """
        return self.get_presigned_url(
            object_name        = asset.minio_key,
            bucket             = asset.minio_bucket,
            expires_in_seconds = expires_in_seconds,
        )

    # ── existence check ────────────────────────────────────────────────────────

    def object_exists(self, object_name: str, bucket: Optional[str] = None) -> bool:
        bucket = bucket or self.MINIO_BUCKET_NAME
        try:
            self.client.stat_object(bucket, object_name)
            return True
        except S3Error:
            return False

    # ── deletion ───────────────────────────────────────────────────────────────

    def delete_object(self, object_name: str, bucket: Optional[str] = None) -> None:
        bucket = bucket or self.MINIO_BUCKET_NAME
        try:
            self.client.remove_object(bucket, object_name)
            with _cache_lock:
                _url_cache.pop((bucket, object_name), None)
            logger.info(f"[MinIO] Deleted {bucket}/{object_name}")
        except S3Error as exc:
            logger.warning(f"[MinIO] Delete failed for {bucket}/{object_name}: {exc}")

    def delete_prefix(self, prefix: str, bucket: Optional[str] = None) -> int:
        """
        Delete ALL objects whose key starts with *prefix*.
        Returns the number of objects deleted.
        Used by DELETE /api/v1/lecture/{stem} to clean up all assets.
        """
        bucket = bucket or self.MINIO_BUCKET_NAME
        deleted = 0
        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
            for obj in objects:
                self.client.remove_object(bucket, obj.object_name)
                with _cache_lock:
                    _url_cache.pop((bucket, obj.object_name), None)
                deleted += 1
            if deleted:
                logger.info(f"[MinIO] Deleted {deleted} object(s) under prefix {bucket}/{prefix}")
        except S3Error as exc:
            logger.error(f"[MinIO] delete_prefix failed for {bucket}/{prefix}: {exc}")
        return deleted

    # ── listing ────────────────────────────────────────────────────────────────

    def list_objects(self, prefix: str = "", bucket: Optional[str] = None, recursive: bool = True):
        bucket = bucket or self.MINIO_BUCKET_NAME
        try:
            return list(self.client.list_objects(bucket, prefix=prefix, recursive=recursive))
        except S3Error as exc:
            logger.error(f"[MinIO] list_objects failed: {exc}")
            return []


# ── singleton ──────────────────────────────────────────────────────────────────
minio_client = MinioClient()