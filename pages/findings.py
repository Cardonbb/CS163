import dash
from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

dash.register_page(__name__, path='/findings', name='Major Findings')

# ---------------------------------------------------------------------------
# Threshold sweep data — all 4 experiments
# ---------------------------------------------------------------------------
EXPERIMENT_DATA = {
    "exp1": {
        "label": "Experiment 1 (Baseline, CE loss, flips only)",
        "rows": [
            # threshold, fault_iou, precision, recall, f1, miou, pixel_acc
            (0.50, 0.2762, 0.2964, 0.8016, 0.4328, 0.5317, 0.8032),
            (0.65, 0.3054, 0.3469, 0.7187, 0.4682, 0.5706, 0.8469),
            (0.70, 0.3167, 0.3741, 0.6735, 0.4811, 0.5857, 0.8639),
            (0.75, 0.3256, 0.4116, 0.6089, 0.4912, 0.6001, 0.8818),
            (0.80, 0.3208, 0.4610, 0.5134, 0.4859, 0.6070, 0.8982),
        ],
    },
    "exp2": {
        "label": "Experiment 2 (CE + Dice + Rotations + Modified LR, Failed)",
        "rows": [
            (0.50, 0.1914, 0.2487, 0.4538, 0.3213, 0.5019, 0.8204),
        ],
    },
    "exp3": {
        "label": "Experiment 3 (Rotations added, Best) ★",
        "rows": [
            (0.50, 0.5518, 0.6864, 0.7377, 0.7111, 0.7457, 0.9439),
            (0.65, 0.5523, 0.7127, 0.7104, 0.7116, 0.7473, 0.9461),
            (0.70, 0.5514, 0.7227, 0.6993, 0.7108, 0.7472, 0.9467),
            (0.75, 0.5496, 0.7339, 0.6864, 0.7093, 0.7466, 0.9461),
            (0.80, 0.3208, 0.4610, 0.5134, 0.4858, 0.6070, 0.8982),
        ],
    },
    "exp4": {
        "label": "Experiment 4 (CE + Dice w=0.1 on Exp 3 config)",
        "rows": [
            (0.50, 0.5147, 0.6200, 0.7380, 0.6738, 0.7254, 0.9423),
            (0.65, 0.5100, 0.6500, 0.7100, 0.6787, 0.7200, 0.9400),
        ],
    },
}

METRIC_COLS = ["fault_iou", "precision", "recall", "f1", "miou", "pixel_acc"]
METRIC_LABELS = {
    "fault_iou": "Fault IoU",
    "precision": "Precision",
    "recall":    "Recall",
    "f1":        "F1 (fault)",
    "miou":      "mIoU",
    "pixel_acc": "Pixel Accuracy",
}


def _experiment_df(key):
    cols = ["threshold"] + METRIC_COLS
    return pd.DataFrame(EXPERIMENT_DATA[key]["rows"], columns=cols)


# ---------------------------------------------------------------------------
# Full experiment comparison — Table 8 from report
# ---------------------------------------------------------------------------
ALL_EXPERIMENTS = [
    # name, fault_iou, bg_iou, miou, f1, precision, recall, pixel_acc, boundary_miou, best_epoch, threshold
    ("Exp 1 (Baseline)",      0.3256, 0.8747, 0.6001, 0.4912, 0.4116, 0.6089, 0.8818, 0.0517, 23,  0.75),
    ("Exp 2 (Failed)",        0.1914, 0.8124, 0.5019, 0.3213, 0.2487, 0.4538, 0.8204, 0.0287, 79,  "N/A"),
    ("Exp 3 (Best) ★",        0.5523, 0.9422, 0.7473, 0.7116, 0.7127, 0.7104, 0.9461, 0.2274, 77,  0.65),
    ("Exp 4 (Dice w=0.1)",    0.5147, 0.9387, 0.7254, 0.6738, 0.7161, 0.6362, 0.9423, 0.2060, 69,  0.65),
]

# ---------------------------------------------------------------------------
# Per-region performance — Experiment 3 test set
# ---------------------------------------------------------------------------
REGION_METRICS = [
    ("Carrizo Plain", 0.7005, 0.8239, 0.7505, 0.9132, 80),
    ("Bay Area",      0.6433, 0.7829, 0.7403, 0.8308, 233),
    ("Mojave",        0.5930, 0.7445, 0.6942, 0.8027, 321),
]

# ---------------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------------
HEADLINE = [
    {"label": "Fault IoU",     "value": "0.55", "sub": "Exp 3, threshold 0.65"},
    {"label": "F1 (fault)",    "value": "0.71", "sub": "Precision 0.71, Recall 0.71"},
    {"label": "Boundary mIoU", "value": "0.23", "sub": "Up from 0.05 at baseline, +344%"},
    {"label": "mIoU",          "value": "0.75", "sub": "Mean of fault and background IoU"},
]

OVERVIEW_MD = """
We ran **four experiments** in a controlled ablation study. The baseline (Exp 1) got the model working
but predictions came out as blobs instead of thin lines. Experiment 2 changed too many things at once
and fell below baseline, which showed us why single-variable testing matters.
Experiment 3 added only rotation augmentation and gave us the biggest jump in the whole project.
Experiment 4 tested Dice loss on top of the best config and it hurt performance every time we tried it.
"""

EXP1_MD = """
**Config:** Cross-entropy loss, class weights [1.0, 8.0], horizontal and vertical flips only.
Early stopping on val/loss, patience 10. Best checkpoint at **epoch 23**.

Recall was 0.80 right away but precision was only 0.30, meaning for every real fault pixel
the model was also flagging about two false positives. Looking at the actual predictions showed
the problem clearly: instead of thin linear traces the model was producing **wide blob-shaped predictions**.
The Boundary mIoU of 0.052 confirmed this since the edges were poorly defined.
Tuning the threshold up to 0.75 pushed Fault IoU from 0.2762 to **0.3256**.
"""

EXP2_MD = """
**Config:** Added Dice loss (w=0.5), raised fault class weight to 15, added rotations,
reduced decoder LR to 1×10⁻⁴, and reduced batch size to 8. Batch size had to drop because
Dice loss increases memory during the backward pass and batch 16 was too large for the A100's 40 GB.

**First attempt:** Collapsed on epoch 1. The model just predicted everything as background,
which gives high pixel accuracy for free since 90%+ of pixels are background anyway.

**Second attempt** (Dice weight reduced to 0.2): Ran all 80 epochs but never actually converged.
The best checkpoint was at epoch 79, which usually means the model was drifting rather than learning.
Fault IoU ended up at **0.1914**, worse than Experiment 1.

**What we learned:** Changing five things at once made it impossible to figure out what caused the drop.
"""

EXP3_MD = """
**Config:** Went back to Experiment 1's exact setup with one change: added 90°, 180°, and 270°
random rotations on top of the existing flips. Early stopping was switched to monitor
val/mIoU with patience 15.

The model trained all the way to **epoch 77** out of 80, which is a big difference from
Experiment 1 peaking at epoch 23. The extra rotation variety gave the model a lot more to
learn from.

At the best threshold of 0.65: **Fault IoU 0.5523**. Boundary mIoU went from 0.052 up to **0.2274**,
which means the predictions went from blobs to actual narrow lines. Precision and recall both
landed at 0.71, so the model stopped being overly aggressive with its predictions.
"""

EXP4_MD = """
**Config:** Used Experiment 3 as the base and added Dice loss at weight 0.1. We tested four
variations: dice w=0.1, dice w=0.2, dice w=0.1 with class weight bumped to 10, and a
standalone fp32 run with dice w=0.1.

Every single configuration did worse than Experiment 3. As we increased the Dice weight
from 0 to 0.2, Fault IoU dropped step by step: 0.5523 → 0.5147 → 0.4282 → 0.3985.
Raising the class weight made it even worse (0.3545).

**Why Dice loss doesn't work here:** Dice loss pushes predictions toward whatever part of the
fault zone the model is most confident about, which ends up being narrower than the 100 m
wide buffer we use as the ground truth label. Also, Prithvi was pretrained using pixel-level
MSE loss, so Dice's region-level objective doesn't match what the encoder was trained to respond to.
"""

KEY_FINDINGS = [
    ("H1: Fine-tuning Prithvi on fault data works.",
     "Before fine-tuning the backbone produced near-empty masks. After training on 4,207 labeled "
     "patches the best model reached Fault IoU 0.55 and F1 0.71. The predictions visibly follow "
     "narrow fault lines rather than just memorizing the background."),
    ("H2: Rotation augmentation was the biggest improvement we made.",
     "With everything else held constant, adding 90/180/270 degree rotations in Experiment 3 "
     "pushed Fault IoU from 0.33 to 0.55, a gain of 0.22 from one config change. "
     "Boundary mIoU jumped from 0.05 to 0.23. The reason it helps so much is that faults run "
     "at every angle in the real world, and Prithvi was only pretrained on north-up images so "
     "it had a built-in orientation bias we needed to correct."),
    ("H3: The model does better in clearer terrain.",
     "Results by region followed the expected order: Carrizo Plain IoU 0.70, "
     "Bay Area 0.64, Mojave 0.59. Carrizo has a very clear San Andreas trace with "
     "high spectral contrast. Mojave has more false positives because dry-wash patterns "
     "and desert pavement look similar to fault scarps in satellite imagery."),
    ("H4: Dice loss made things worse on our buffered labels.",
     "Every Dice configuration in Experiment 4 performed worse than Experiment 3. "
     "The core issue is that Dice loss narrows predictions toward the high-confidence center "
     "of the fault zone, but our labels are a 100 m wide buffer so the model ends up "
     "predicting a thinner zone than the ground truth. Weighted cross-entropy works better here."),
    ("Testing one thing at a time matters.",
     "Experiment 2 changed five settings at once and dropped below baseline (IoU 0.19). "
     "Experiment 3 changed only one thing and produced the biggest gain in the project. "
     "That result shaped all of our later decisions."),
]

LIMITATIONS_MD = """
- **Coverage:** We only labeled about 300 km² across three regions. California is 423,970 km² total,
  so we haven't tested how well this generalizes to places like the Central Valley or the North Coast.
- **Label quality:** The USGS fault database is the best source we have but it's not complete.
  The exact position of fault traces at 10 m resolution has some uncertainty, which is part of why
  we used the 50 m buffer.
- **Optical imagery only:** Roads, field boundaries, and vegetation edges can look similar to fault
  traces in satellite imagery. Without elevation data it's hard to separate them in some areas.
- **Resolution:** The model predicts a 100 m wide fault zone, not a single centerline. To use
  this for mapping you'd still need a post-processing step to extract the actual fault trace.
"""

FUTURE_MD = """
1. **Train longer:** Experiment 3 peaked at epoch 77 out of 80, so it probably hadn't fully converged.
   Training to 120 epochs could squeeze out more performance.
2. **Add an elevation model branch:** Train a second model on 1-meter LiDAR DEM data
   (hillshade, slope, aspect, roughness) to detect faults from elevation alone.
3. **Combine optical and elevation:** Fuse the Sentinel-2 model and DEM model at the feature level.
   Roads are flat in elevation while fault scarps have a clear topographic signature, so combining
   the two should reduce false positives.
4. **Have geologists review the errors:** A lot of what looks like a false positive to the model
   might actually be an unmapped fault. Having domain experts go through the misclassifications
   would help separate real errors from label gaps.
5. **Try a topology-aware loss:** clDice computes overlap along the fault centerline skeleton
   rather than the buffered zone. That's a better fit for thin linear features and worth testing.
6. **Run it on all of California:** Apply the trained model statewide at 10 m resolution to
   generate a candidate fault list for geologist review.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metric_card(label, value, sub):
    return dbc.Card(
        dbc.CardBody([
            html.Div(value, className="display-6 fw-bold text-primary"),
            html.Div(label, className="fw-semibold"),
            html.Div(sub, className="text-muted small"),
        ]),
        className="text-center h-100 shadow-sm",
    )


def _section(heading, *body):
    return html.Div([
        html.H2(heading, className="mt-5 mb-3"),
        *body,
    ])


# ---------------------------------------------------------------------------
# Full experiment comparison table
# ---------------------------------------------------------------------------

def _all_experiments_table():
    header = html.Thead(html.Tr([
        html.Th("Experiment"), html.Th("Fault IoU"), html.Th("mIoU"),
        html.Th("F1"), html.Th("Precision"), html.Th("Recall"),
        html.Th("Boundary mIoU"), html.Th("Best Epoch"), html.Th("Threshold"),
    ]))
    rows = []
    for exp in ALL_EXPERIMENTS:
        name, fault_iou, bg_iou, miou, f1, prec, rec, pix, bmIoU, epoch, thr = exp
        is_best = "★" in name
        style = {"backgroundColor": "#eafaf1"} if is_best else {}
        rows.append(html.Tr([
            html.Td(html.Strong(name) if is_best else name),
            html.Td(f"{fault_iou:.4f}"),
            html.Td(f"{miou:.4f}"),
            html.Td(f"{f1:.4f}"),
            html.Td(f"{prec:.4f}"),
            html.Td(f"{rec:.4f}"),
            html.Td(f"{bmIoU:.4f}"),
            html.Td(str(epoch)),
            html.Td(str(thr)),
        ], style=style))
    return dbc.Table([header, html.Tbody(rows)], bordered=True, hover=True,
                     responsive=True, size="sm")


# ---------------------------------------------------------------------------
# Per-region figure + table
# ---------------------------------------------------------------------------

def _region_metrics_fig():
    names = [r[0] for r in REGION_METRICS]
    metrics = {
        "IoU":       [r[1] for r in REGION_METRICS],
        "F1":        [r[2] for r in REGION_METRICS],
        "Precision": [r[3] for r in REGION_METRICS],
        "Recall":    [r[4] for r in REGION_METRICS],
    }
    colors = {"IoU": "#c0392b", "F1": "#1f77b4",
              "Precision": "#2ca02c", "Recall": "#ff7f0e"}
    fig = go.Figure()
    for name, vals in metrics.items():
        fig.add_trace(go.Bar(name=name, x=names, y=vals,
                             marker_color=colors[name],
                             text=[f"{v:.2f}" for v in vals],
                             textposition="outside"))
    fig.update_layout(
        title="Experiment 3 — test set metrics by region",
        yaxis_range=[0, 1.05], yaxis_title="Score",
        barmode="group", height=420,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return fig


def _region_metrics_table():
    header = html.Thead(html.Tr([
        html.Th("Region"), html.Th("IoU"), html.Th("F1"),
        html.Th("Precision"), html.Th("Recall"), html.Th("Test patches"),
    ]))
    body = html.Tbody([
        html.Tr([html.Td(r[0]), html.Td(f"{r[1]:.4f}"), html.Td(f"{r[2]:.4f}"),
                 html.Td(f"{r[3]:.4f}"), html.Td(f"{r[4]:.4f}"), html.Td(str(r[5]))])
        for r in REGION_METRICS
    ])
    return dbc.Table([header, body], bordered=True, hover=True,
                     responsive=True, size="sm")


# ---------------------------------------------------------------------------
# Interactive threshold sweep
# ---------------------------------------------------------------------------
interactive = html.Div([
    dbc.Row([
        dbc.Col([
            html.Label("Experiment", className="fw-semibold"),
            dcc.RadioItems(
                id='findings-experiment',
                options=[{"label": EXPERIMENT_DATA[k]["label"], "value": k}
                         for k in EXPERIMENT_DATA],
                value="exp3", inline=False,
                inputClassName="me-2", labelClassName="d-block mb-1",
            ),
        ], md=7),
        dbc.Col([
            html.Label("Threshold", className="fw-semibold"),
            dcc.Slider(
                id='findings-threshold',
                min=0.50, max=0.80, step=0.05, value=0.65,
                marks={x / 100: f"{x/100:.2f}" for x in range(50, 81, 5)},
            ),
        ], md=5),
    ], className="mb-4"),

    dcc.Graph(id='findings-sweep-graph'),
    dbc.Row(id='findings-metric-row', className="g-3 mt-2"),
])


@callback(
    Output('findings-sweep-graph', 'figure'),
    Output('findings-metric-row', 'children'),
    Input('findings-experiment', 'value'),
    Input('findings-threshold', 'value'),
)
def update_sweep(exp_key, threshold):
    df = _experiment_df(exp_key)
    fig = go.Figure()
    for col in METRIC_COLS:
        fig.add_trace(go.Scatter(
            x=df["threshold"], y=df[col],
            mode="lines+markers", name=METRIC_LABELS[col],
        ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="gray",
                  annotation_text=f"threshold = {threshold:.2f}",
                  annotation_position="top")
    fig.update_layout(
        title=f"{EXPERIMENT_DATA[exp_key]['label']}: metrics vs threshold",
        xaxis_title="Classification threshold (probability cutoff for 'fault')",
        yaxis_title="Metric value", yaxis=dict(range=[0, 1]),
        hovermode="x unified", height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    nearest_idx = (df["threshold"] - threshold).abs().idxmin()
    row = df.iloc[nearest_idx]
    cards = [
        dbc.Col(_metric_card(METRIC_LABELS[col], f"{row[col]:.4f}",
                              f"@ thr {row['threshold']:.2f}"), md=2)
        for col in METRIC_COLS
    ]
    return fig, cards


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = dbc.Container([

    html.Div([
        html.P("RESULTS", className="text-uppercase text-muted small mb-2"),
        html.H1("Major Findings", className="display-5 fw-bold mb-3"),
        html.P("What worked, what failed, and what the numbers mean.",
               className="lead text-muted"),
    ], className="py-4"),

    dbc.Row(
        [dbc.Col(_metric_card(**h), md=3) for h in HEADLINE],
        className="g-3 mb-5",
    ),

    _section("Overview", dcc.Markdown(OVERVIEW_MD)),

    _section(
        "All Experiments Comparison",
        html.P(
            "Full metrics from all four experiments on the held-out test set (634 patches). "
            "Experiment 3 achieves best performance on every primary metric. "
            "Green row = best model.",
            className="text-muted small",
        ),
        _all_experiments_table(),
    ),

    _section(
        "Interactive: Threshold Sweep",
        dcc.Markdown(
            "Pick an experiment and slide the **classification threshold**, which is the probability "
            "cutoff for calling a pixel 'fault'. Moving it up reduces false positives "
            "but misses more real faults. Moving it down catches more faults but also picks up more noise."
        ),
        interactive,
    ),

    _section("Experiment 1: Baseline", dcc.Markdown(EXP1_MD)),
    _section("Experiment 2: Failed Run", dcc.Markdown(EXP2_MD)),
    _section("Experiment 3: Rotation Augmentation (Best)", dcc.Markdown(EXP3_MD)),
    _section("Experiment 4: Dice Loss Ablation", dcc.Markdown(EXP4_MD)),

    _section(
        "Per-Region Performance (Experiment 3)",
        dcc.Markdown(
            "Test-set metrics broken out by region. Carrizo Plain scored highest because the "
            "San Andreas trace there is very exposed with strong spectral contrast. "
            "Mojave scored lowest because desert pavement and dry-wash channels look similar to fault scarps."
        ),
        dcc.Graph(figure=_region_metrics_fig(), config={"displayModeBar": False}),
        _region_metrics_table(),
    ),

    _section(
        "Qualitative Results",
        html.P(
            "Test patches from the Experiment 3 checkpoint at threshold 0.65. "
            "Each column shows the Sentinel-2 RGB input, the USGS ground truth mask (50 m buffer), "
            "and the model prediction.",
            className="text-muted small",
        ),
        html.Img(
            src=dash.get_asset_url("e3.png"),
            className="img-fluid rounded shadow-sm mt-2",
            alt="Experiment 3 test predictions",
        ),
        html.H5("Fault Line Overlay", className="mt-4 mb-2 fw-semibold"),
        html.P(
            "The same predictions with the actual USGS fault centerline drawn on top in red. "
            "This shows how well the predicted zone lines up with the real fault location.",
            className="text-muted small",
        ),
        html.Img(
            src=dash.get_asset_url("line.png"),
            className="img-fluid rounded shadow-sm mt-2",
            alt="Experiment 3 predictions with fault centerline overlay",
        ),
    ),

    _section(
        "Key Findings",
        html.Ul([
            html.Li([html.Strong(t), " ", html.Span(d)])
            for t, d in KEY_FINDINGS
        ]),
    ),

    _section("Limitations", dcc.Markdown(LIMITATIONS_MD)),

    _section("Future Work", dcc.Markdown(FUTURE_MD)),

    html.Div(className="py-5"),

], fluid=False)
