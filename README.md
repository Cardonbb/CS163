# FaultFinder — Active Fault Detection with Prithvi-EO 2.0

A fine-tuned geospatial foundation model for detecting active geological fault traces from Sentinel-2 satellite imagery. Built for CS163 at UCSC.

## Overview

FaultFinder fine-tunes the [Prithvi-EO 2.0](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL) 600M-parameter Vision Transformer (IBM/NASA) on a novel dataset of 4,207 labeled 128×128 patches across three California regions. The model performs semantic segmentation to identify fault-zone pixels in 6-band Sentinel-2 imagery.

**Best result (Experiment 3):** Fault IoU 0.5523 · F1 0.7116 · mIoU 0.7473

## Architecture

- **Encoder:** Prithvi-EO-2.0-600M-TL (ViT-H, frozen pretrained weights, fine-tuned)
- **Decoder:** UperNet with FPN + PPM, 256 channels, 2-class head
- **Input:** 6-band Sentinel-2 (B2, B3, B4, B8, B11, B12) at 10 m/pixel, 128×128 patches upsampled to 224×224
- **Framework:** TerraTorch + PyTorch Lightning

## Dataset

Three California training regions covering diverse environments:

| Region | Environment | Key Faults | Patches |
|--------|-------------|------------|---------|
| Carrizo Plain | Mediterranean shrubland | Central San Andreas | 531 |
| Mojave Desert | Arid desert | S. San Andreas, Mojave | 2,141 |
| Bay Area | Urban/oak woodland | Hayward, Calaveras | 1,535 |

Labels are derived from the USGS Quaternary Fault and Fold Database, buffered 50 m (5 px) to 100-meter-wide fault zones. Total: 4,207 patches (70/15/15 train/val/test split stratified by region).

## Key Findings

1. **Rotation augmentation** (+90°/180°/270°) was the single largest performance driver: **+0.2267 fault IoU** over the baseline. Prithvi-EO 2.0's sinusoidal positional encodings are not rotation-invariant, and HLS pretraining used exclusively north-up images — augmentation during fine-tuning is required to learn orientation-invariant fault representations.

2. **Dice loss degrades performance** on buffered linear labels. Dice's region-size bias pushes predictions narrower than the annotation buffer, and its global overlap objective conflicts with Prithvi's pixel-level MSE pretraining. All four Dice configurations tested underperformed weighted cross-entropy.

3. **Controlled ablation is critical.** Experiment 2 (multi-variable change) failed and could not be diagnosed. Experiment 3 (single-variable change) immediately identified rotations as the key driver.

## Experiments

| Experiment | Fault IoU | Notes |
|------------|-----------|-------|
| Exp 1 — CE loss, flip augmentation | 0.3256 | Baseline; blob predictions |
| Exp 2 — CE+Dice, rotations, modified LR | 0.1914 | Multi-variable change; inconclusive |
| **Exp 3 — CE loss, rotations** | **0.5523** | Best; single-variable improvement |
| Exp 4 — Dice ablation (w=0.1) | 0.5147 | Consistent degradation vs. Exp 3 |

## Project Structure

```
cs163/
├── dashapps/           # Plotly Dash visualization apps
│   ├── app1.py – app8.py
│   └── app5-multi/     # Multi-page app (home + analytics)
├── assets/
│   └── eda_pics/       # EDA visualizations (region maps, sample patches)
├── train_colab*.ipynb  # Training notebook (Google Colab / A100)
├── INFERENCE.md        # End-to-end inference pipeline guide
├── report.md           # Full paper / technical report
├── Dockerfile
└── requirements.txt
```

## Setup

```bash
pip install terratorch torch torchvision lightning rasterio numpy matplotlib \
            earthengine-api geopandas dash plotly gunicorn
```

For inference, also authenticate Google Earth Engine:

```bash
earthengine authenticate
```

## Running the Dashboard

```bash
cd dashapps
python app1.py          # individual apps
# or
python app5-multi/app5.py   # multi-page app
```

Or via Docker:

```bash
docker build -t faultfinder .
docker run -p 8080:8080 faultfinder
```

## Inference

See [INFERENCE.md](INFERENCE.md) for the full pipeline: downloading Sentinel-2 imagery via GEE, tiling into 128×128 patches, running model inference, and stitching predictions into a georeferenced GeoTIFF.

```bash
python infer_download.py --name my_region --bbox 200000 4050000 400000 4200000
python infer_tile.py     --image data/inference/my_region_10m.tif --out data/inference/patches/my_region/
python infer_run.py      --patches data/inference/patches/my_region/ --checkpoint checkpoints/prithvi600m-fault-best.ckpt --out outputs/inference/my_region/
python infer_stitch.py   --patches outputs/inference/my_region/raw/ --image data/inference/my_region_10m.tif --out outputs/inference/my_region/fault_probability.tif
```

Default binary threshold: **0.65** (tuned on test set). Lower to 0.3 for exploration; raise to 0.7 for high-confidence mapping.

## Training Configuration

- Optimizer: AdamW — encoder LR 5e-5, decoder LR 5e-4 (10× ratio)
- Scheduler: CosineAnnealingLR → 1e-7 over 80 epochs
- Batch size: 16 · Precision: fp32
- Hardware: NVIDIA A100-SXM4-40GB (Google Colab)
- Loss: Weighted cross-entropy, class weights [1.0, 8.0]
- Augmentation: horizontal flip, vertical flip, 90°/180°/270° rotation

## Limitations

- Trained on ~300 km² across three California environments (0.07% of the state)
- Road networks and agricultural field boundaries cause systematic false positives in optical imagery alone — LiDAR DEM fusion (planned) is expected to resolve this
- Predictions are 100-meter-wide fault zones; centerline extraction requires post-processing skeletonization

## Citation

If you use this work, please cite the project report:

> FaultFinder: Active Fault Detection from Sentinel-2 Imagery using Prithvi-EO 2.0. CS163, UC Santa Cruz, 2025.
