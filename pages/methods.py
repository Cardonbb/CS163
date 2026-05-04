import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import math

dash.register_page(__name__, path='/methods', name='Analysis Methods')

INTRO_MD = """
This page walks through every analysis step that turns raw Sentinel-2 imagery and
USGS fault vectors into a fine-tuned segmentation model. Each subsection covers
one method: what it does, why it is needed, and the figure or table that summarizes it.
"""

PREPROCESSING_MD = """
Sentinel-2 Level-2A scenes were pulled through the **Google Earth Engine (GEE)** Python API
for each of the three study regions, drawing from imagery acquired between
**June 2022 and September 2023**. Summer months were selected to minimize cloud cover
and maximize soil and rock exposure in California's seasonally dry climate.

Cloud masking used the **Sentinel-2 Scene Classification Layer (SCL)**, masking
cloud shadow (class 3), medium probability cloud (class 8), high probability cloud (class 9),
and cirrus (class 10). Remaining valid observations were combined into a per-pixel
median composite to suppress shadows and seasonal vegetation.

Images were exported at **10-meter spatial resolution** in UTM Zone 10N (EPSG:32610),
comprising six surface-reflectance bands: Blue (B2), Green (B3), Red (B4),
Near Infrared (B8), SWIR1 (B11), and SWIR2 (B12). These are the same six bands
used in Prithvi-EO 2.0 pretraining, ensuring spectral alignment with the pretrained encoder.

USGS Quaternary Fault and Fold vector traces were filtered to active California faults,
**buffered 50 m** on each side to absorb digitizing error, and rasterized to align
pixel-for-pixel with the imagery.
"""

PATCHING_MD = """
The aligned imagery and label rasters were chipped into **128×128 patches** using a
sliding window with a **64-pixel (50%) stride**. At 10-meter resolution, each patch
covers 1.28 km × 1.28 km of terrain — sufficient to capture the linear continuity of
fault traces and surrounding context while remaining computationally tractable.

Two quality filters were applied:
- Patches with more than **50% NoData** pixels (satellite orbit boundary artifacts) were discarded.
- Patches with fewer than **0.5% fault pixels** were discarded to prevent the dataset from
  being dominated by fault-free background examples.

After filtering, **4,207 patches** were retained (8.9% retention rate) and divided into
train (2,943), val (630), and test (634) splits by spatial chunks so that nearby
patches do not leak across splits.
"""

CLASS_IMBALANCE_MD = """
Even after the 0.5% filter, fault pixels make up roughly **1.2% of the total**.
We use **weighted cross-entropy** with w_background = 1.0 and w_fault = 8.0 so the
loss does not collapse to "predict background everywhere."

An earlier experiment (Experiment 2) pushed the fault weight to 15 combined with
Dice loss — that run never converged, which is why we settled on a single conservative
class weight of 8.0.
"""

AUGMENTATION_MD = """
Faults run at every angle, so showing the model only horizontal/vertical flips
leaves most orientations underrepresented. The final augmentation set is:

- Horizontal and vertical flips (p = 0.5 each)
- 90°, 180°, and 270° rotations (uniform over the four orientations)

**Why rotations matter for Prithvi:** Despite using self-attention, Prithvi-EO 2.0 is
not rotation-invariant. The model uses 3D sinusoidal positional encodings that hard-code
absolute spatial coordinates into each patch token. A 90° rotation produces a fundamentally
different token sequence even though the visual content is identical. Furthermore, HLS
pretraining was conducted on north-up images exclusively — no statistical rotation
invariance was acquired during pretraining. Rotation augmentation during fine-tuning
is therefore the correct mechanism to achieve the needed invariance.

Adding the three rotations was the **single biggest jump in the project** — Fault IoU
moved from 0.33 to 0.55 with no other config change.
"""

ARCHITECTURE_MD = """
The backbone is **Prithvi-EO-2.0-600M** (ViT-H architecture), pretrained by IBM, NASA,
and the Jülich Supercomputing Centre via masked autoencoder (MAE) self-supervision on
4.2 million HLS global time-series samples. Key features include 3D patch embeddings,
3D positional encodings, geolocation embeddings (latitude, longitude), and temporal
embeddings (year, day-of-year).

For our single-timestamp application we set the temporal dimension **T=1** and supply
geolocation coordinates per patch. The model operates at a native resolution of 224×224 px.
We upsample 128×128 input patches to 224×224 using bilinear interpolation prior to encoding.

Multi-scale features are extracted from transformer layers at indices **[7, 15, 23, 31]**
— four evenly spaced layers spanning the full 32-layer depth — using the TerraTorch
SelectIndices neck followed by ReshapeTokensToImage to recover spatial structure
from token sequences.

**Total parameter count: 643,294,532**
- Encoder (Prithvi backbone): 631,188,482 parameters
- Decoder head (UperNet): 12,106,050 parameters
- All parameters are trained — no frozen layers.
"""

DECODER_MD = """
We use **UperNet** (Unified Perceptual Parsing Network) as the segmentation decoder,
configured via TerraTorch's EncoderDecoderFactory. UperNet aggregates multi-scale
features through a Feature Pyramid Network (FPN) and Pyramid Pooling Module (PPM),
producing a rich multi-resolution feature representation upsampled to the input
resolution and projected to class probabilities by a convolutional head.

Decoder configuration:
- **256** feature channels
- Two-layer classification head: **[128, 64]** channels
- **Dropout 0.1** applied in the classification head
"""

TRAINING_MD = """
All experiments use **AdamW** with differential learning rates:
- **Encoder LR: 5×10⁻⁵** — conservative to preserve pretrained representations
- **Decoder LR: 5×10⁻⁴** — 10× higher because the decoder starts from random weights

A **CosineAnnealingLR** schedule decays both learning rates to **1×10⁻⁷** over 80 epochs
with a 3-epoch linear warm-up. Weight decay of **0.1** is applied throughout.

Training uses **batch size 16** on an NVIDIA A100-SXM4-40GB GPU (40 GB VRAM) with
full precision (fp32). Early stopping monitors **val/mIoU** with patience 15
(val/loss with patience 10 in Experiment 1). The best checkpoint by the monitored
metric is used for all reported test evaluations.
"""

EVALUATION_MD = """
Cross-entropy gives every pixel a fault probability. We sweep the classification
threshold over **[0.50, 0.65, 0.70, 0.75, 0.80]** and select the threshold that
maximizes fault-class IoU.

Given the asymmetric cost structure — a missed fault carries far greater consequence
for seismic hazard than a false positive — we accept precision-recall tradeoffs that
favor recall.
"""

DICE_LOSS_MD = """
Experiment 4 systematically tested Dice loss on top of the Experiment 3 configuration.
All four Dice configurations degraded performance **monotonically** — the more Dice weight,
the worse the result. This is not a tuning failure; it has a principled theoretical explanation.

**Region-size bias:** Dice loss pulls predictions toward the most-confident sub-region of
the foreground. For buffered fault labels — where the 100 m wide zone is an artificial
annotation device — this causes the model to predict narrower than the target width even
when the label is wide. Edge pixels (which have lower image evidence than the centerline)
receive small, noisy Dice gradients, and the optimum lies well inside the buffer.

**Pretraining mismatch:** Prithvi-EO 2.0 was pretrained with pixel-level MSE reconstruction
loss. Dice loss optimizes a global region-overlap objective — a fundamental mismatch with
the diffuse pixel-level gradients that conditioned the pretrained encoder. Cross-entropy's
pixel-level gradients are better aligned with the pretraining objective for transfer learning.

**Implication:** For any geospatial segmentation task where ground truth is generated by
buffering thin features (roads, rivers, fault traces, shorelines), Dice loss is likely to
underperform weighted cross-entropy. The topology-aware alternative — if shape precision
is critical — is **clDice**, computed against the unbuffered centerline skeleton rather
than the buffered mask.
"""


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _class_weight_fig():
    fig = go.Figure(go.Bar(
        x=["Background", "Fault"],
        y=[1.0, 8.0],
        marker_color=["#9aa0a6", "#c0392b"],
        text=["1.0", "8.0"],
        textposition="outside",
    ))
    fig.update_layout(
        title="Cross-entropy class weights",
        yaxis_title="Weight", yaxis_range=[0, 10],
        height=300, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _augmentation_fig():
    angles_flip = [0, 90, 180, 270]
    coverage_flip = [1, 0, 1, 0]
    coverage_rot  = [1, 1, 1, 1]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Flips only",
                         x=[f"{a}°" for a in angles_flip], y=coverage_flip,
                         marker_color="#9aa0a6"))
    fig.add_trace(go.Bar(name="Flips + rotations",
                         x=[f"{a}°" for a in angles_flip], y=coverage_rot,
                         marker_color="#c0392b"))
    fig.update_layout(
        title="Orientation coverage after augmentation",
        yaxis=dict(visible=False, range=[0, 1.2]),
        barmode="group", height=300,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return fig


def _lr_schedule_fig():
    epochs = list(range(0, 81))
    enc_peak, dec_peak = 5e-5, 5e-4
    warmup = 3
    total = 80

    def cosine(peak, e):
        if e < warmup:
            return peak * (e + 1) / warmup
        progress = (e - warmup) / max(1, total - warmup)
        return peak * 0.5 * (1 + math.cos(math.pi * progress))

    enc = [cosine(enc_peak, e) for e in epochs]
    dec = [cosine(dec_peak, e) for e in epochs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=dec, mode="lines",
                             name="Decoder LR (5×10⁻⁴)", line=dict(color="#c0392b", width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=enc, mode="lines",
                             name="Encoder LR (5×10⁻⁵)", line=dict(color="#1f77b4", width=2)))
    fig.update_layout(
        title="Cosine LR schedule — both decay to 1×10⁻⁷ (3-epoch warm-up)",
        xaxis_title="Epoch", yaxis_title="Learning rate",
        height=320, margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return fig


def _dice_comparison_fig():
    configs = ["Exp 3\n(no Dice)", "Dice w=0.1\n(standalone)", "Dice w=0.1\n(sweep)", "Dice w=0.2\n(sweep)", "Dice w=0.1\nw_cls=10"]
    fault_iou = [0.5523, 0.5147, 0.4282, 0.3985, 0.3545]
    colors = ["#2ca02c"] + ["#c0392b"] * 4
    fig = go.Figure(go.Bar(
        x=configs, y=fault_iou,
        marker_color=colors,
        text=[f"{v:.4f}" for v in fault_iou],
        textposition="outside",
    ))
    fig.update_layout(
        title="Dice loss consistently degrades Fault IoU (Experiment 4)",
        yaxis_title="Fault IoU", yaxis_range=[0, 0.65],
        height=360, margin=dict(l=40, r=20, t=60, b=60),
    )
    return fig


def _metrics_table():
    rows = [
        ("Fault IoU",       "TP / (TP + FP + FN) for the fault class only.",
         "Primary metric — selects the operating threshold."),
        ("Precision",       "Of pixels predicted as fault, how many are correct.",
         "Trustworthiness of a positive prediction."),
        ("Recall",          "Of real fault pixels, how many are recovered.",
         "How much of the true trace the model finds."),
        ("F1 (fault)",      "Harmonic mean of fault precision and recall.",
         "Single-number summary at the operating threshold."),
        ("mIoU",            "Mean IoU across fault and background classes.",
         "Cross-comparison with other segmentation papers."),
        ("Boundary mIoU",   "IoU restricted to a narrow band around the trace.",
         "Penalises blob predictions that miss the actual fault line."),
        ("Pixel Accuracy",  "Fraction of pixels classified correctly overall.",
         "Reported for completeness — dominated by background class."),
    ]
    header = html.Thead(html.Tr([
        html.Th("Metric"), html.Th("Definition"), html.Th("Why we report it"),
    ]))
    body = html.Tbody([
        html.Tr([html.Td(html.Strong(m)),
                 html.Td(d, className="small"),
                 html.Td(w, className="small text-muted")])
        for m, d, w in rows
    ])
    return dbc.Table([header, body], bordered=True, hover=True,
                     responsive=True, className="align-top")


REFERENCES = [
    {"label": "Prithvi-EO-2.0 (NASA + IBM + Jülich)",
     "url": "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
     "note": "Foundation model used as the encoder backbone."},
    {"label": "USGS Quaternary Fault and Fold Database",
     "url": "https://www.usgs.gov/programs/earthquake-hazards/faults",
     "note": "Source of fault labels."},
    {"label": "Sentinel-2 mission",
     "url": "https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2",
     "note": "10 m multispectral imagery."},
    {"label": "TerraTorch",
     "url": "https://github.com/IBM/terratorch",
     "note": "Framework used to fine-tune Prithvi."},
    {"label": "UperNet (Xiao et al., 2018)",
     "url": "https://arxiv.org/abs/1807.10221",
     "note": "Decoder architecture."},
    {"label": "Kervadec et al. (2024) — Dice loss decomposition",
     "url": "https://arxiv.org/abs/2104.12099",
     "note": "Theoretical basis for Dice loss region-size bias analysis."},
    {"label": "clDice (Shit et al., 2021)",
     "url": "https://arxiv.org/abs/2003.07311",
     "note": "Topology-aware loss for thin linear structures."},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(heading, *body):
    return html.Div([
        html.H2(heading, className="mt-5 mb-3"),
        *body,
    ])


def _reference_item(label, url, note):
    return html.Li([
        html.A(label, href=url, target="_blank"),
        html.Span(f" — {note}", className="text-muted") if note else None,
    ])


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = dbc.Container([

    html.Div([
        html.P("APPROACH", className="text-uppercase text-muted small mb-2"),
        html.H1("Analysis Methods", className="display-5 fw-bold mb-3"),
        html.P("How the data is built and how the model is trained — one method "
               "per subsection.", className="lead text-muted"),
        dcc.Markdown(INTRO_MD),
    ], className="py-4"),

    _section("Preprocessing", dcc.Markdown(PREPROCESSING_MD)),

    _section("Patch Sampling", dcc.Markdown(PATCHING_MD)),

    _section("Class Imbalance",
             dcc.Markdown(CLASS_IMBALANCE_MD),
             dcc.Graph(figure=_class_weight_fig(),
                       config={"displayModeBar": False})),

    _section("Augmentation",
             dcc.Markdown(AUGMENTATION_MD),
             dcc.Graph(figure=_augmentation_fig(),
                       config={"displayModeBar": False})),

    _section("Model Architecture — Encoder (Prithvi-EO 2.0)",
             dcc.Markdown(ARCHITECTURE_MD)),

    _section("Model Architecture — Decoder (UperNet)",
             dcc.Markdown(DECODER_MD)),

    _section("Training Setup",
             dcc.Markdown(TRAINING_MD),
             dcc.Graph(figure=_lr_schedule_fig(),
                       config={"displayModeBar": False})),

    _section("Evaluation Metrics",
             dcc.Markdown(EVALUATION_MD),
             _metrics_table()),

    _section("References & Tools",
             html.Ul([_reference_item(**r) for r in REFERENCES])),

    html.Div(className="py-5"),

], fluid=False)
