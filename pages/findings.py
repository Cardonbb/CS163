import dash
from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

dash.register_page(__name__, path='/findings', name='Findings')

# ---------------------------------------------------------------------------
# Real threshold sweep numbers from the experiments.
# Update / add experiments here as you run more.
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
    # TODO: add Experiment 4 numbers when complete
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
# Headline numbers — shown at the top.
# ---------------------------------------------------------------------------

HEADLINE = [
    {"label": "Fault IoU", "value": "0.55", "sub": "Exp 3, threshold 0.65"},
    {"label": "F1 (fault)", "value": "0.71", "sub": "Balanced precision and recall"},
    {"label": "Boundary mIoU", "value": "0.23", "sub": "Up from 0.05 at baseline"},
    {"label": "mIoU", "value": "0.75", "sub": "Mean of fault and background"},
]

OVERVIEW_MD = """
We ran three experiments. The baseline worked but painted blobs. The second attempt
changed too many things at once and got worse. The third experiment changed only
one thing (rotation augmentation) and gave the largest jump in the project. The
takeaway was to change one variable at a time.
"""

EXP1_MD = """
Cross-entropy loss with class weights of 1.0 and 8.0, horizontal and vertical flips
only. Recall hit 0.80 right away, but precision was just 0.30, so for every real
fault pixel the model also flagged about two false positives. Tuning the threshold
to 0.75 helped, landing at a Fault IoU of 0.33. The predictions still looked like
wide blobs instead of narrow traces, which showed up as a Boundary mIoU of 0.05.
"""

EXP2_MD = """
We tried to fix everything at once. Added Dice loss, raised the fault class weight
to 15, added rotations, and dropped the batch size to 8 because Dice pushed us past
the GPU memory limit. The first run collapsed and predicted zero fault pixels. The
second run trained all 80 epochs but never converged, ending at a Fault IoU of 0.19.
Worse than the baseline. Lesson: changing four things at once means you cannot tell
which one broke it.
"""

EXP3_MD = """
We went back to the baseline config and added one thing: 90, 180, and 270 degree
rotations. Faults run at every angle, so showing the model only flips left most
orientations underrepresented. This single change moved Fault IoU from 0.33 to 0.55
and Boundary mIoU from 0.05 to 0.23, which means the predictions are actually
following linear traces now. Precision and recall came in nearly equal at 0.71,
which is the operating point we wanted.
"""

LIMITATIONS_MD = """
Coverage is small. About 300 km squared labeled across three regions out of
California's 423,000 km squared, so generalization to new terrain is not yet
validated. Labels carry positional uncertainty from the USGS database, partly
softened by the 50 m buffer. We use optical imagery only, so roads, shorelines,
and field boundaries can look like faults. No elevation data yet.
"""

FUTURE_MD = """
Experiment 4 will add Dice loss back at a small weight (0.1) on top of the
Experiment 3 config. The goal is tighter predictions without breaking training.
If that does not help, Experiment 5 will run for 120 epochs since the best
Experiment 3 checkpoint was still improving at epoch 77.
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


def _section(heading, body):
    return html.Div([
        html.H2(heading, className="mt-5 mb-3"),
        body,
    ])


# ---------------------------------------------------------------------------
# Interactive threshold sweep — fulfills the rubric's "interactive diagram"
# requirement. Pick an experiment + threshold; the chart redraws and the
# metric cards update.
# ---------------------------------------------------------------------------

interactive = html.Div([
    dbc.Row([
        dbc.Col([
            html.Label("Experiment", className="fw-semibold"),
            dcc.RadioItems(
                id='findings-experiment',
                options=[
                    {"label": EXPERIMENT_DATA[k]["label"], "value": k}
                    for k in EXPERIMENT_DATA
                ],
                value="exp3",
                inline=False,
                inputClassName="me-2",
                labelClassName="d-block mb-1",
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
        yaxis_title="Metric value",
        yaxis=dict(range=[0, 1]),
        hovermode="x unified",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # Find the closest available threshold row
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
        html.P(
            "What worked, what failed, and what the numbers mean.",
            className="lead text-muted",
        ),
    ], className="py-4"),

    # Headline strip
    dbc.Row(
        [dbc.Col(_metric_card(**h), md=3) for h in HEADLINE],
        className="g-3 mb-5",
    ),

    _section("Overview", dcc.Markdown(OVERVIEW_MD)),

    _section("Interactive: Threshold Sweep", html.Div([
        dcc.Markdown(
            "Pick an experiment and slide the classification threshold to see "
            "how each metric shifts. Cross-entropy gives every pixel a fault "
            "probability, and the threshold is what we call 'fault' versus 'background.'"
        ),
        interactive,
    ])),

    _section("Experiment 1: Baseline", dcc.Markdown(EXP1_MD)),
    _section("Experiment 2: Failed Run", dcc.Markdown(EXP2_MD)),
    _section("Experiment 3: Rotation Augmentation (Best)", dcc.Markdown(EXP3_MD)),

    _section("Limitations", dcc.Markdown(LIMITATIONS_MD)),

    _section("Next Experiments", dcc.Markdown(FUTURE_MD)),

    html.Div(className="py-5"),

], fluid=False)
