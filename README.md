# FaultFinder — Active Fault Detection with Prithvi-EO 2.0

A fine-tuned geospatial foundation model for detecting active geological fault traces from Sentinel-2 satellite imagery. Built for CS163 at UCSC.

**Live website:** https://cs63-494919.uw.r.appspot.com/

---

## Overview

FaultFinder fine-tunes the [Prithvi-EO 2.0](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL) 600M-parameter Vision Transformer (IBM/NASA) on a novel dataset of 4,207 labeled 128×128 patches across three California regions. The model performs semantic segmentation to identify fault-zone pixels in 6-band Sentinel-2 imagery.

**Best result (Experiment 3):** Fault IoU 0.5523 · F1 0.7116 · mIoU 0.7473

---

## Pipeline: Data Collection → Analysis → Website

```
1. DATA COLLECTION
   Google Earth Engine (GEE) → Sentinel-2 median composites (.tif)
   USGS Quaternary Fault DB  → fault vector labels (.shp)
        │
        ▼
2. PREPROCESSING  [train.ipynb / inference/]
   Reproject to UTM Zone 10N → buffer fault lines 50 m
   Tile into 128×128 patches (.npy) → quality filter → train/val/test split
        │
        ▼
3. MODEL TRAINING  [train.ipynb]
   Fine-tune Prithvi-EO 2.0 + UperNet on A100 (Google Colab)
   Ablation study across 4 experiments → best checkpoint saved (.ckpt)
        │
        ▼
4. INFERENCE  [inference/ + Dockerfile]
   Load checkpoint → run over all test patches
   Precompute fault probability maps → cache results
   Deploy containerized inference service to Google Cloud Run
        │
        ▼
5. CLOUD STORAGE  [cloud_io.py]
   Upload Sentinel-2 imagery + patch .npy files
   to GCS bucket  gs://cs163-fault-data-carter/
        │
        ▼
6. WEBSITE PUBLICATION  [app1.py + pages/ + app.yaml]
   Dash app deployed to Google App Engine
   Reads data from GCS at runtime (local fallback for dev)
   Calls Cloud Run inference endpoint for predictions
   → https://cs63-494919.uw.r.appspot.com/
```

---

## Repository Structure

```
cs163/
│
├── app1.py              # Main Dash app entry point (served by gunicorn)
├── cloud_io.py          # GCS loader with local fallback; used by all pages
├── app.yaml             # Google App Engine config (runtime, scaling, env vars)
├── Dockerfile           # Container definition for Cloud Run inference service
├── requirements.txt     # Python dependencies
├── train.ipynb          # Full model training pipeline (Colab / A100)
├── INFERENCE.md         # Step-by-step guide for running statewide inference
│
├── pages/               # Multi-page Dash route handlers (EDA, Methods,
│                        #   Findings, etc.) — each file is one website page
│
├── dashapps/            # Standalone Dash prototypes used during development
│   ├── app1.py – app8.py   individual component experiments
│   └── app5-multi/         multi-page layout prototype
│
├── inference/           # Inference pipeline scripts
│                        #   (download → tile → run → stitch)
│
├── data/                # Local copies of dataset metadata and patch files
│                        #   (mirrored from GCS; used as offline fallback)
│
└── assets/
    └── eda_pics/        # Pre-generated EDA images embedded in the website
                         #   (region maps, sample patches, mask previews)
```

---

## System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │  HTTPS
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│          Google App Engine  (app.yaml / app1.py)                │
│          runtime: python310  •  gunicorn -b :8080 app1:server   │
│          auto-scaling: target CPU 90%, max 1 instance           │
│                                                                  │
│   Dash app renders pages, charts, and fault prediction maps      │
│            │                          │                          │
│       reads metadata               calls inference               │
└────────────┼──────────────────────────┼─────────────────────────┘
             │                          │
             ▼                          ▼
┌────────────────────────┐   ┌──────────────────────────────────┐
│  Google Cloud Storage  │   │  Google Cloud Run                │
│  gs://cs163-fault-     │   │  fault-inference-*.us-west1      │
│    data-carter/        │   │  run.app                         │
│                        │   │                                  │
│  • Sentinel-2 .tif     │   │  Containerized Prithvi-EO 2.0    │
│  • Patch .npy files    │   │  model serving precomputed        │
│  • dataset_info.json   │   │  fault predictions               │
│  • CA outline          │   │                                  │
└────────────────────────┘   └──────────────────────────────────┘
```

### Scalability

| Component | Current setup | Scales by |
|-----------|--------------|-----------|
| App Engine (website) | max 1 instance, auto-scales to CPU 90% | Raise `max_instances` in `app.yaml` |
| Cloud Run (inference) | serverless, scales to zero | Increases replicas automatically on load |
| GCS (storage) | object storage, no instance limit | Inherently horizontally scalable |

Inference results are **precomputed and cached** — the Cloud Run service is called once per region, not per user request. This means the website serves fast static results regardless of concurrent users, and Cloud Run GPU costs are not incurred per page load.

---

## Inference Service

**Docker code:** [`Dockerfile`](Dockerfile)  
**Model training code:** [`train.ipynb`](train.ipynb)  
**Inference pipeline:** [`inference/`](inference/) and [`INFERENCE.md`](INFERENCE.md)

The Cloud Run service wraps the trained Prithvi-EO 2.0 checkpoint in a containerized REST API. It accepts Sentinel-2 patch arrays and returns per-pixel fault probability maps.

| | Detail |
|-|--------|
| **Endpoint** | `https://fault-inference-489370349569.us-west1.run.app` |
| **Input** | 6-band Sentinel-2 patch, shape `[1, 1, 6, 128, 128]` (batch × time × channels × H × W) |
| **Output** | Fault probability map, shape `[1, 2, 128, 128]` — class 0 = background, class 1 = fault |
| **Post-processing** | `softmax(output)[:, 1, :, :]` → fault probability per pixel; threshold at 0.65 for binary mask |
| **Container** | Python 3.10 + gunicorn, built from [`Dockerfile`](Dockerfile), served on port 8080 |

Predictions are precomputed across all test patches and stored in GCS. The Dash website reads these cached results rather than calling the endpoint live per request.

---

## Cloud Storage

**Bucket:** `gs://cs163-fault-data-carter/`  
**Access code:** [`cloud_io.py`](cloud_io.py)

| Stored object | Path in bucket | Consumed by |
|--------------|---------------|-------------|
| Sentinel-2 imagery | `imagery/<region>_10m.tif` | Inference pipeline, EDA page |
| Patch `.npy` files | `patches/<region>/` | Inference pipeline |
| Dataset metadata | `dataset/dataset_info.json` | EDA and Methods pages |
| California outline | `dataset/ca_outline.*` | Map visualizations |

The website reads from GCS at runtime via `cloud_io.py`, which tries the GCS bucket first and transparently falls back to local copies under `data/` if GCS is unreachable (e.g., during local development). Files are fetched using the `google-cloud-storage` Python client and cached in-process with `lru_cache`.

---

## Architecture

- **Encoder:** Prithvi-EO-2.0-600M-TL (ViT-H, fine-tuned)
- **Decoder:** UperNet with FPN + PPM, 256 channels, 2-class head
- **Input:** 6-band Sentinel-2 (B2, B3, B4, B8, B11, B12) at 10 m/pixel, 128×128 patches upsampled to 224×224
- **Framework:** TerraTorch + PyTorch Lightning

---

## Dataset

Three California training regions covering diverse environments:

| Region | Environment | Key Faults | Patches |
|--------|-------------|------------|---------|
| Carrizo Plain | Mediterranean shrubland | Central San Andreas | 531 |
| Mojave Desert | Arid desert | S. San Andreas, Mojave | 2,141 |
| Bay Area | Urban/oak woodland | Hayward, Calaveras | 1,535 |

Labels derived from the USGS Quaternary Fault and Fold Database, buffered 50 m to 100-meter-wide fault zones. Total: 4,207 patches (70/15/15 train/val/test split stratified by region).

---

## Experiments

| Experiment | Fault IoU | Notes |
|------------|-----------|-------|
| Exp 1 — CE loss, flip augmentation | 0.3256 | Baseline; blob predictions |
| Exp 2 — CE+Dice, rotations, modified LR | 0.1914 | Multi-variable change; inconclusive |
| **Exp 3 — CE loss, rotations** | **0.5523** | Best; single-variable improvement |
| Exp 4 — Dice ablation (w=0.1) | 0.5147 | Consistent degradation vs. Exp 3 |

**Key findings:**
1. Rotation augmentation (+90°/180°/270°) produced a **+0.2267 fault IoU** gain — the single largest improvement. Prithvi-EO 2.0's sinusoidal positional encodings are not rotation-invariant, and HLS pretraining used exclusively north-up images.
2. Dice loss consistently degrades performance on buffered linear labels due to its region-size bias and mismatch with Prithvi's pixel-level MSE pretraining objective.

---

## Training Configuration

- Optimizer: AdamW — encoder LR 5e-5, decoder LR 5e-4 (10× ratio)
- Scheduler: CosineAnnealingLR → 1e-7 over 80 epochs
- Batch size: 16 · Precision: fp32
- Hardware: NVIDIA A100-SXM4-40GB (Google Colab)
- Loss: Weighted cross-entropy, class weights [1.0, 8.0]
- Augmentation: horizontal flip, vertical flip, 90°/180°/270° rotation

---

## Setup

```bash
pip install terratorch torch torchvision lightning rasterio numpy matplotlib \
            earthengine-api geopandas dash plotly gunicorn google-cloud-storage
```

For inference, also authenticate Google Earth Engine:

```bash
earthengine authenticate
```

Run the website locally:

```bash
python app1.py
```

Or via Docker:

```bash
docker build -t faultfinder .
docker run -p 8080:8080 faultfinder
```

---

## Limitations

- Trained on ~300 km² across three California environments (0.07% of the state)
- Road networks and agricultural field boundaries cause systematic false positives in optical imagery — LiDAR DEM fusion is planned to resolve this
- Predictions are 100-meter-wide fault zones; centerline extraction requires post-processing skeletonization
