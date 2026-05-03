import dash
from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

dash.register_page(__name__, path='/findings', name='Major Findings')

# ---------------------------------------------------------------------------
# Real threshold sweep numbers from the experiments.
# ---------------------------------------------------------------------------
EXPERIMENT_DATA = {
    "exp1": {
        "label": "Experiment 1: Baseline (CE loss, flips only)",
        "rows": [
            # threshold, fault_iou, precision, recall, f1, miou, pixel_acc
            (0.50, 0.2762, 0.2964, 0.8016, 0.4328, 0.5317, 0.8032),
            (0.65, 0.3054, 0.3469, 0.7187, 0.4682, 0.5706, 0.8469),
            (0.70, 0.3167, 0.3741, 0.6735, 0.4811, 0.5857, 0.8639),
            (0.75, 0.3256, 0.4116, 0.6089, 0.4912, 0.6001, 0.8818),
            (0.80, 0.3208, 0.4610, 0.5134, 0.4859, 0.6070, 0.8982),
        ],
    },
    "exp3": {
        "label": "Experiment 3: + Rotations (best)",
        "rows": [
            (0.50, 0.5518, 0.6864, 0.7377, 0.7111, 0.7457, 0.9439),
            (0.65, 0.5523, 0.7127, 0.7104, 0.7116, 0.7473, 0.9461),
            (0.70, 0.5514, 0.7227, 0.6993, 0.7108, 0.7472, 0.9467),
            (0.75, 0.5496, 0.7339, 0.6864, 0.7093, 0.7466, 0.9461),
        ],
    },
}

METRIC_COLS = ["fault_iou", "precision", "recall", "f1", "miou", "pixel_acc"]
METRIC_LABELS = {
    "fault_iou": "Fault IoU",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 (fault)",
    "miou": "mIoU",
    "pixel_acc": "Pixel Accuracy",
}


def _experiment_df(key):
    cols = ["threshold"] + METRIC_COLS
    return pd.DataFrame(EXPERIMENT_DATA[key]["rows"], columns=cols)


# ---------------------------------------------------------------------------
# Per-region performance — from the Experiment 3 evaluation cell.
# bay_area  IoU=0.6433  F1=0.7829  P=0.7403  R=0.8308  (n=248)
# carrizo   IoU=0.7005  F1=0.8239  P=0.7505  R=0.9132  (n=68)
# mojave    IoU=0.5930  F1=0.7445  P=0.6942  R=0.8027  (n=316)
# ---------------------------------------------------------------------------
REGION_METRICS = [
    ("Carrizo Plain", 0.7005, 0.8239, 0.7505, 0.9132, 68),
    ("Bay Area",      0.6433, 0.7829, 0.7403, 0.8308, 248),
    ("Mojave",        0.5930, 0.7445, 0.6942, 0.8027, 316),
]

# ---------------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------------
HEADLINE = [
    {"label": "Fault IoU",     "value": "0.55", "sub": "Exp 3, threshold 0.65"},
    {"label": "F1 (fault)",    "value": "0.71", "sub": "Balanced precision and recall"},
    {"label": "Boundary mIoU", "value": "0.23", "sub": "Up from 0.05 at baseline"},
    {"label": "mIoU",          "value": "0.75", "sub": "Mean of fault and background"},
]

OVERVIEW_MD = """
We ran three experiments. The baseline worked but painted blobs. The second
attempt changed too many things at once and got worse. The third experiment
changed only one thing — rotation augmentation — and gave the largest jump in the
project. The takeaway is to change one variable at a time.
"""

EXP1_MD = """
Cross-entropy loss with class weights of 1.0 and 8.0, horizontal and vertical
flips only. Recall hit 0.80 right away, but precision was just 0.30, so for every
real fault pixel the model also flagged about two false positives. Tuning the
threshold to 0.75 helped, landing at a Fault IoU of 0.33. The predictions still
looked like wide blobs instead of narrow traces, which showed up as a Boundary
mIoU of 0.05.
"""

EXP2_MD = """
We tried to fix everything at once. Added Dice loss, raised the fault class
weight to 15, added rotations, and dropped the batch size to 8 because Dice
pushed us past the GPU memory limit. The first run collapsed and predicted zero
fault pixels. The second run trained all 80 epochs but never converged, ending
at a Fault IoU of 0.19. Worse than the baseline. Lesson: changing four things at
once means you cannot tell which one broke it.
"""

EXP3_MD = """
We went back to the baseline config and added one thing: 90°, 180°, and 270°
rotations. Faults run at every angle, so showing the model only flips left most
orientations underrepresented. This single change moved Fault IoU from 0.33 to
0.55 and Boundary mIoU from 0.05 to 0.23, which means the predictions are
actually following linear traces now. Precision and recall came in nearly equal
at 0.71, which is the operating point we wanted.
"""

KEY_FINDINGS = [
    ("One variable at a time wins.",
     "Experiment 2 changed loss, weights, augmentation, and batch size all at "
     "once and regressed. Experiment 3 changed only the augmentation set and "
     "gave the biggest jump in the project."),
    ("Rotation augmentation matters more than loss tweaks.",
     "Faults have no preferred orientation, so flips alone leave the model "
     "blind to two of the four cardinal directions. Adding 90/180/270° "
     "rotations improved Fault IoU by 22 points."),
    ("Boundary mIoU tracks visual quality better than pixel accuracy.",
     "Pixel accuracy was already 0.90 at baseline because the dataset is "
     "98.8% background. Boundary mIoU jumped from 0.05 to 0.23 between "
     "Exp 1 and Exp 3, which actually reflects the predictions now "
     "following linear traces."),
    ("Carrizo Plain is the easiest region; Mojave is the hardest.",
     "Carrizo IoU = 0.70, Bay Area = 0.64, Mojave = 0.59. Carrizo's "
     "exposed San Andreas trace has high contrast against the surrounding "
     "rock; Mojave's desert pavement looks similar to fault scarps in "
     "places."),
]

LIMITATIONS_MD = """
Coverage is small — about 300 km² labeled across three regions out of California's
423,000 km², so generalization to new terrain is not yet validated. Labels carry
positional uncertainty from the USGS database, partly softened by the 50 m
buffer. We use optical imagery only, so roads, shorelines, and field boundaries
can occasionally look like faults. No elevation data has been integrated yet.
"""

FUTURE_MD = """
Experiment 4 will add Dice loss back at a small weight (0.1) on top of the
Experiment 3 config — the goal is tighter predictions without breaking training.
If that does not help, Experiment 5 will run for 120 epochs since the best
Experiment 3 checkpoint was still improving at epoch 77. Longer term, adding a
DEM-derived slope channel should help disambiguate fault scarps from roads.
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
        title="Experiment 3 metrics by region (test set)",
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
        html.Th("Precision"), html.Th("Recall"), html.Th("n (test patches)"),
    ]))
    body = html.Tbody([
        html.Tr([html.Td(r[0]), html.Td(f"{r[1]:.4f}"),
                 html.Td(f"{r[2]:.4f}"), html.Td(f"{r[3]:.4f}"),
                 html.Td(f"{r[4]:.4f}"), html.Td(str(r[5]))])
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
        ], md=6),
        dbc.Col([
            html.Label("Threshold", className="fw-semibold"),
            dcc.Slider(
                id='findings-threshold',
                min=0.50, max=0.80, step=0.05, value=0.65,
                marks={x / 100: f"{x/100:.2f}" for x in range(50, 81, 5)},
            ),
        ], md=6),
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
        xaxis_title="Classification threshold",
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

    # Headline strip
    dbc.Row(
        [dbc.Col(_metric_card(**h), md=3) for h in HEADLINE],
        className="g-3 mb-5",
    ),

    _section("Overview", dcc.Markdown(OVERVIEW_MD)),

    _section(
        "Interactive: Threshold Sweep",
        dcc.Markdown(
            "Pick an experiment and slide the classification threshold to see "
            "how each metric shifts. Cross-entropy gives every pixel a fault "
            "probability, and the threshold is what we call 'fault' versus "
            "'background.'"
        ),
        interactive,
    ),

    _section("Experiment 1: Baseline", dcc.Markdown(EXP1_MD)),
    _section("Experiment 2: Failed Run", dcc.Markdown(EXP2_MD)),
    _section("Experiment 3: Rotation Augmentation (Best)", dcc.Markdown(EXP3_MD)),

    _section(
        "Per-Region Performance",
        dcc.Markdown(
            "Test-set metrics from the Experiment 3 checkpoint, broken out by "
            "region. Carrizo Plain is the easiest — its exposed San Andreas "
            "trace has high contrast against the surrounding rock. Mojave is "
            "hardest because desert pavement and dry-wash patterns can mimic "
            "fault scarps."
        ),
        dcc.Graph(figure=_region_metrics_fig(),
                  config={"displayModeBar": False}),
        _region_metrics_table(),
    ),

    _section(
        "Qualitative Results",
        dcc.Markdown(
            "Six held-out test tiles from the best Experiment 3 checkpoint. "
            "**Top:** Sentinel-2 RGB input. **Middle:** USGS ground-truth fault "
            "traces (50 m buffer). **Bottom:** model prediction at threshold "
            "0.65. The model recovers narrow linear traces rather than the "
            "wide blobs seen at baseline."
        ),
        html.Div([
            html.Div([
                html.Div("Sentinel-2 RGB",
                         className="fw-semibold text-end pe-2"),
                html.Div("Ground Truth",
                         className="fw-semibold text-end pe-2"),
                html.Div("Model Prediction",
                         className="fw-semibold text-end pe-2"),
            ], style={"display": "flex", "flexDirection": "column",
                       "justifyContent": "space-around",
                       "minWidth": "140px", "fontSize": "0.95rem"}),
            html.Img(src=dash.get_asset_url("test_predictions_cropped.png"),
                     className="img-fluid rounded shadow-sm",
                     style={"flex": "1 1 auto", "minWidth": 0},
                     alt="Test predictions: Sentinel-2 image, ground truth, "
                         "and model prediction for six tiles."),
        ], className="d-flex align-items-stretch mt-3"),
    ),

    _section(
        "Key Findings",
        html.Ul([
            html.Li([html.Strong(t), " — ", html.Span(d)])
            for t, d in KEY_FINDINGS
        ]),
    ),

    _section("Limitations", dcc.Markdown(LIMITATIONS_MD)),

    _section("Next Experiments", dcc.Markdown(FUTURE_MD)),

    html.Div(className="py-5"),

], fluid=False)
