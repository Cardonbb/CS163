"""Cloud Run inference service for CS163 fault detection.

Endpoints
---------
GET /health                     — liveness check
GET /patches                    — list patch names from the test split
GET /predict?patch=<patch_name> — return satellite RGB, ground truth, and
                                  model prediction as base64 PNGs
"""

from __future__ import annotations

import base64
import io
import logging
import os
from functools import lru_cache

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BUCKET = os.environ.get("CS163_BUCKET", "cs163-fault-data-carter")


@lru_cache(maxsize=1)
def _gcs_client():
    from google.cloud import storage  # type: ignore
    return storage.Client()


def _read_bytes(path: str) -> bytes:
    return _gcs_client().bucket(BUCKET).blob(path).download_as_bytes()


def _load_npy(path: str) -> np.ndarray:
    return np.load(io.BytesIO(_read_bytes(path)))


def _to_png_b64(arr: np.ndarray) -> str:
    """Encode a uint8 H×W×3 array as a base64 PNG string."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _build_rgb(img: np.ndarray) -> np.ndarray:
    """Convert [6, H, W] satellite array to uint8 [H, W, 3] RGB.

    Band order is [Blue, Green, Red, NIR, SWIR1, SWIR2] so indices 2,1,0
    give Red, Green, Blue. We percentile-stretch to enhance contrast.
    """
    rgb = img[[2, 1, 0], :, :].transpose(1, 2, 0).astype(np.float32)
    lo, hi = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb * 255).astype(np.uint8)


def _overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Paint mask=1 pixels red on a copy of the RGB image."""
    out = rgb.copy()
    out[mask == 1] = [220, 30, 30]
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/patches")
def patches():
    try:
        blobs = _gcs_client().list_blobs(BUCKET, prefix="predictions/")
        names = [
            b.name.removeprefix("predictions/").removesuffix(".npy")
            for b in blobs
            if b.name.endswith(".npy")
        ]
        return jsonify({"patches": sorted(names), "count": len(names)})
    except Exception as exc:
        logger.error("Could not list predictions: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.get("/predict")
def predict():
    """Return satellite RGB, ground-truth overlay, and prediction overlay.

    Query params
    ------------
    patch : str  e.g. ``mojave_r01856_c03136``

    Returns JSON
    ------------
    patch, rgb_png, ground_truth_png, prediction_png (all base64),
    fault_pixel_fraction, ground_truth_fraction
    """
    patch = request.args.get("patch", "").strip()
    if not patch:
        return jsonify({"error": "patch query parameter is required"}), 400

    try:
        img = _load_npy(f"dataset/images/{patch}.npy")
    except Exception as exc:
        return jsonify({"error": f"image not found: {exc}"}), 404

    try:
        label = _load_npy(f"dataset/labels/{patch}.npy").astype(np.uint8)
    except Exception as exc:
        return jsonify({"error": f"label not found: {exc}"}), 404

    try:
        pred = _load_npy(f"predictions/{patch}.npy").astype(np.uint8)
    except Exception as exc:
        return jsonify({"error": f"prediction not found: {exc}"}), 404

    rgb = _build_rgb(img)
    h, w = rgb.shape[:2]

    # Resize masks to match the satellite image dimensions
    def _resize_mask(mask: np.ndarray) -> np.ndarray:
        return np.array(
            Image.fromarray(mask * 255).resize((w, h), Image.NEAREST)
        ) // 255

    label_r = _resize_mask(label)
    pred_r  = _resize_mask(pred)

    return jsonify({
        "patch": patch,
        "rgb_png":           _to_png_b64(rgb),
        "ground_truth_png":  _to_png_b64(_overlay(rgb, label_r)),
        "prediction_png":    _to_png_b64(_overlay(rgb, pred_r)),
        "fault_pixel_fraction":  float(pred.mean()),
        "ground_truth_fraction": float(label.mean()),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
