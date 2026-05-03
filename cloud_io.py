"""GCS-backed loader with local fallback.

The Dash site reads a few small metadata files (California outline, dataset
splits, dataset_info.json) at import time. In production those live in
gs://cs163-fault-data-carter/; locally we fall back to the bundled copies under
data/ and contextOfProject/patches/ so the site still runs offline.

Usage:
    from cloud_io import load_json, load_text, GCS_SOURCE
    info = load_json("dataset/dataset_info.json", local="contextOfProject/patches/dataset_info.json")
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

BUCKET_NAME = os.environ.get("CS163_BUCKET", "cs163-fault-data-carter")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Set to "gcs" or "local" after the first successful read so the EDA / Methods
# pages can show users where data is being served from.
GCS_SOURCE = "unknown"


@lru_cache(maxsize=1)
def _client():
    """Lazily build a GCS client. Fails silently for offline / no-auth dev."""
    try:
        from google.cloud import storage  # type: ignore
        return storage.Client()
    except Exception as exc:  # pragma: no cover - offline path
        logger.warning("GCS client unavailable, will use local fallback: %s", exc)
        return None


def _read_gcs_bytes(path: str) -> Optional[bytes]:
    """Fetch path from gs://BUCKET_NAME/<path>. Returns None on any failure."""
    client = _client()
    if client is None:
        return None
    try:
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(path)
        return blob.download_as_bytes()
    except Exception as exc:  # network / permission / missing — treat as miss
        logger.warning("GCS read failed for gs://%s/%s: %s",
                       BUCKET_NAME, path, exc)
        return None


def _read_local_bytes(local: str) -> bytes:
    abs_path = local if os.path.isabs(local) else os.path.join(PROJECT_ROOT, local)
    with open(abs_path, "rb") as f:
        return f.read()


def load_bytes(gcs_path: str, local: str) -> bytes:
    """Try GCS first; fall back to a local file shipped in the deploy bundle."""
    global GCS_SOURCE
    data = _read_gcs_bytes(gcs_path)
    if data is not None:
        GCS_SOURCE = "gcs"
        return data
    GCS_SOURCE = "local"
    return _read_local_bytes(local)


def load_text(gcs_path: str, local: str, encoding: str = "utf-8") -> str:
    return load_bytes(gcs_path, local).decode(encoding)


def load_json(gcs_path: str, local: str):
    return json.loads(load_text(gcs_path, local))


def gcs_uri(path: str) -> str:
    return f"gs://{BUCKET_NAME}/{path}"
