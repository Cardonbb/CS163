import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import math

dash.register_page(__name__, path='/methods', name='Analysis Methods')

# ---------------------------------------------------------------------------
# Page intro
# ---------------------------------------------------------------------------
INTRO_MD = """
This page walks through every analysis step that turns raw Sentinel-2 imagery and
USGS fault vectors into a fine-tuned segmentation model. Each subsection covers
one method: what it does, why it is needed, and the figure or table that
summarizes it.
"""

# ---------------------------------------------------------------------------
# Subsection bodies
# ---------------------------------------------------------------------------
PREPROCESSING_MD = """
Sentinel-2 Level-2A scenes are pulled through Earth Engine for each of the three
study regions, masked with the SCL cloud band, and reduced to a dry-season median
composite to suppress shadows and seasonal vegetation. We keep six surface-
reflectance bands (Blue, Green, Red, NIR, SWIR1, SWIR2) at native 10 m resolution.
USGS Quaternary Fault and Fold vector traces are filtered to active California
faults, buffered 50 m on each side to absorb digitizing error, and rasterized to
align pixel-for-pixel with the imagery.
"""

PATCHING_MD = """
The aligned imagery and label rasters are chipped into 128×128 patches with a
sliding window of stride 64 (50% overlap). Patches with no fault pixels are
dropped, and any patch with less than 0.5% fault pixels is also dropped — without
this filter the dataset would be dominated by empty background. The result is
4,207 patches split into train (2,944), val (631), and test (632) by spatial
chunks so that nearby patches do not leak across splits.
"""

CLASS_IMBALANCE_MD = """
Even after the 0.5% filter, fault pixels are roughly 1.2% of the total. We use
weighted cross-entropy with **w_background = 1.0** and **w_fault = 8.0** so the
loss does not collapse to "predict background everywhere". An earlier experiment
pushed the fault weight to 15 with Dice loss; that run never converged, which is
why we settled on a single conservative class weight.
"""

AUGMENTATION_MD = """
Faults run at every angle, so showing the model only horizontal/vertical flips
leaves most orientations underrepresented. The final augmentation set is:

- Horizontal and vertical flips (50% each)
- 90°, 180°, and 270° rotations (uniform over the four orientations)

Adding the three rotations was the single biggest jump in the project — Fault IoU
moved from 0.33 to 0.55 with no other config change.
"""

ARCHITECTURE_MD = """
The backbone is **Prithvi-EO-2.0** at 600M parameters, pretrained by NASA and IBM
on 4.2 million satellite scenes. We attach a **UperNet** decoder for binary
segmentation. Multi-scale features are pulled from encoder layers 7, 15, 23, and
31; the decoder runs at 256 channels with a head that steps down to 128 then 64
before the final 1×1 logit conv.
"""

TRAINING_MD = """
Training runs up to 80 epochs at batch size 16 on a single A100. The encoder
uses a learning rate of 5×10⁻⁵; the decoder runs at 1×10⁻⁴ — 10× higher because
it starts from random weights. We use cosine annealing with a 3-epoch warm-up and
weight decay 0.1. Early stopping watches validation mIoU with a patience of 15.
"""

EVALUATION_MD = """
Cross-entropy gives every pixel a fault probability. We sweep the classification
threshold from 0.50 to 0.80 in 0.05 steps and report Fault IoU, Precision,
Recall, F1, mean IoU (mIoU), Boundary mIoU, and pixel accuracy at every step.
The operating threshold is the one that maximises Fault IoU — for the best run
this is 0.65.
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
    # Show how rotation augmentation flips vs full-rotation set covers orientations.
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
    enc_peak, dec_peak = 5e-5, 1e-4
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
                             name="Decoder LR", line=dict(color="#c0392b", width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=enc, mode="lines",
                             name="Encoder LR", line=dict(color="#1f77b4", width=2)))
    fig.update_layout(
        title="Cosine learning-rate schedule (3-epoch warm-up)",
        xaxis_title="Epoch", yaxis_title="Learning rate",
        height=320, margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return fig


def _metrics_table():
    rows = [
        ("Fault IoU",      "Intersection-over-union on the fault class only.",
         "Primary headline metric — selects the operating threshold."),
        ("Precision",      "Of pixels predicted as fault, how many are correct.",
         "Tells you how trustworthy a positive prediction is."),
        ("Recall",         "Of real fault pixels, how many are recovered.",
         "Tells you how much of the true trace the model is missing."),
        ("F1 (fault)",     "Harmonic mean of fault precision and recall.",
         "Single number summary at the operating threshold."),
        ("mIoU",           "Mean IoU across fault and background classes.",
         "Cross-comparison with other segmentation papers."),
        ("Boundary mIoU",  "mIoU restricted to a narrow band around the trace.",
         "Penalises blob predictions that miss the actual fault line."),
        ("Pixel accuracy", "Fraction of pixels classified correctly overall.",
         "Reported for completeness — dominated by background."),
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
    {"label": "Prithvi-EO-2.0 (NASA + IBM)",
     "url": "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
     "note": "Foundation model used as the backbone."},
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

    _section("Model Architecture", dcc.Markdown(ARCHITECTURE_MD)),

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
