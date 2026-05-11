"""DGX Flask inference server for CS163 fault detection.

On DGX:
    cd ~/cs163
    conda activate prithvi
    export GOOGLE_APPLICATION_CREDENTIALS=$HOME/cs163/.secrets/dgx-inference.json
    CUDA_VISIBLE_DEVICES=0 python server.py

Access via Tailscale: http://100.107.21.115:8000
"""
from __future__ import annotations

import base64
import io
import logging
import os
from functools import lru_cache

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BUCKET  = os.environ.get("CS163_BUCKET", "cs163-fault-data-carter")
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
CKPT    = os.path.expanduser("~/cs163/checkpoints/mIoU=0.7427.ckpt")
THRESHOLD = 0.65

HLS_MEANS = np.array([0.2450, 0.2683, 0.2772, 0.5264, 0.5096, 0.3233], dtype=np.float32)
HLS_STDS  = np.array([0.1367, 0.1292, 0.1273, 0.1571, 0.1231, 0.1201], dtype=np.float32)

# ---------------------------------------------------------------------------
# Model — loaded once at startup
# ---------------------------------------------------------------------------

def _load_model():
    from terratorch.tasks import SemanticSegmentationTask

    logger.info(f"Loading checkpoint {CKPT} on {DEVICE} ...")
    try:
        task = SemanticSegmentationTask.load_from_checkpoint(
            CKPT, map_location=DEVICE, strict=False
        )
        logger.info("Loaded via load_from_checkpoint")
    except Exception as e:
        logger.warning(f"load_from_checkpoint failed ({e}), using manual rebuild")
        ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
        hp   = ckpt["hyper_parameters"]
        hp["model_args"]["backbone_kwargs"]["pretrained"] = False  # skip 2.6 GB HF download
        task = SemanticSegmentationTask(
            model_args     = hp["model_args"],
            model_factory  = hp["model_factory"],
            loss           = hp["loss"],
            class_weights  = hp.get("class_weights"),
            ignore_index   = hp.get("ignore_index", -1),
            freeze_backbone= False,
            freeze_decoder = False,
        )
        task.load_state_dict(ckpt["state_dict"], strict=False)
        logger.info("Loaded via manual rebuild")

    return task.to(DEVICE).eval()


MODEL = _load_model()
logger.info(f"Model ready on {DEVICE}.")

# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _gcs():
    from google.cloud import storage
    return storage.Client()


def _read_npy(blob_path: str) -> np.ndarray:
    data = _gcs().bucket(BUCKET).blob(blob_path).download_as_bytes()
    return np.load(io.BytesIO(data))

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _build_rgb(img: np.ndarray) -> np.ndarray:
    """[6, H, W] → uint8 [H, W, 3] RGB with percentile stretch."""
    rgb = img[[2, 1, 0]].transpose(1, 2, 0).astype(np.float32)
    lo, hi = np.percentile(rgb, [2, 98])
    return np.clip((rgb - lo) / (hi - lo + 1e-8) * 255, 0, 255).astype(np.uint8)


def _overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    mask_r = np.array(
        Image.fromarray(mask * 255).resize((w, h), Image.NEAREST)
    ) // 255
    out = rgb.copy()
    out[mask_r == 1] = [220, 30, 30]
    return out


def _png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _preprocess(img: np.ndarray) -> torch.Tensor:
    """[6, 128, 128] → [1, 6, 1, 224, 224] normalized tensor on DEVICE."""
    img = img.astype(np.float32)
    np.nan_to_num(img, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, 0.0, 1.0)
    img = (img - HLS_MEANS[:, None, None]) / HLS_STDS[:, None, None]
    t = torch.from_numpy(img).unsqueeze(0)                           # [1, 6, 128, 128]
    t = torch.nn.functional.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    return t.unsqueeze(2).to(DEVICE)                                  # [1, 6, 1, 224, 224]

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify({"status": "ok", "device": DEVICE})


@app.get("/patches")
def patches():
    try:
        blobs = _gcs().list_blobs(BUCKET, prefix="predictions/")
        names = sorted(
            b.name.removeprefix("predictions/").removesuffix(".npy")
            for b in blobs if b.name.endswith(".npy")
        )
        return jsonify({"patches": names, "count": len(names)})
    except Exception as exc:
        logger.error("Could not list patches: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.get("/predict")
def predict():
    patch = request.args.get("patch", "").strip()
    if not patch:
        return jsonify({"error": "patch parameter required"}), 400

    try:
        img = _read_npy(f"dataset/images/{patch}.npy")
    except Exception as exc:
        return jsonify({"error": f"image not found: {exc}"}), 404

    try:
        label = _read_npy(f"dataset/labels/{patch}.npy").astype(np.uint8)
    except Exception:
        label = None

    rgb = _build_rgb(img)
    x   = _preprocess(img)

    with torch.no_grad():
        out    = MODEL(x)
        logits = out.output if hasattr(out, "output") else out
        prob   = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()  # [224, 224]

    pred = (prob > THRESHOLD).astype(np.uint8)

    result = {
        "patch":                patch,
        "rgb_png":              _png_b64(rgb),
        "prediction_png":       _png_b64(_overlay(rgb, pred)),
        "fault_pixel_fraction": float(pred.mean()),
    }
    if label is not None:
        result["ground_truth_png"]      = _png_b64(_overlay(rgb, label))
        result["ground_truth_fraction"] = float(label.mean())

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
