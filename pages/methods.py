import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/methods', name='Methods')

# ---------------------------------------------------------------------------
# This page can be more technical than the others. Each PIPELINE_STEPS entry
# is one row in the methods table — the same shape as your Analysis Method doc.
# ---------------------------------------------------------------------------

PIPELINE_OVERVIEW_MD = """
Pull Sentinel-2 tiles for three California regions, mask clouds, take a median
composite, and rasterize USGS fault vectors into binary masks. Chip everything into
128 by 128 pixel patches. Fine-tune Prithvi-EO-2.0 with a UperNet decoder. Sweep
the classification threshold and report metrics.
"""

PIPELINE_STEPS = [
    {
        "name": "Fetch imagery",
        "input": "Region of interest, date range",
        "output": "Sentinel-2 median composite (6 bands)",
        "purpose": "Get clean cloud-free imagery to feed the model.",
        "how": "Earth Engine query, SCL cloud mask, dry-season window for rock and soil contrast.",
    },
    {
        "name": "Build labels",
        "input": "USGS Quaternary Fault and Fold Database",
        "output": "Binary fault raster at 10 m",
        "purpose": "Turn vector fault traces into pixel labels the model can learn from.",
        "how": "Filter to active California faults, buffer 50 m on each side, rasterize.",
    },
    {
        "name": "Chip patches",
        "input": "Aligned imagery and label rasters",
        "output": "4,207 patches at 128 by 128 pixels",
        "purpose": "Make the dataset trainable on a single GPU.",
        "how": "Sliding window with overlap. Drop patches that contain no fault pixels at all.",
    },
    {
        "name": "Fine-tune",
        "input": "Patches plus Prithvi-EO-2.0 weights",
        "output": "Trained checkpoint",
        "purpose": "Teach the foundation model to look for fault landforms.",
        "how": "TerraTorch, UperNet decoder, class-weighted cross entropy, rotation augmentation.",
    },
    {
        "name": "Evaluate",
        "input": "Held-out test patches",
        "output": "Fault IoU, F1, Boundary mIoU at multiple thresholds",
        "purpose": "Pick the operating threshold and report honest numbers.",
        "how": "Sweep threshold from 0.50 to 0.80, report all metrics, choose the peak Fault IoU.",
    },
]

MODEL_MD = """
The backbone is Prithvi-EO-2.0 at 600M parameters, pretrained by NASA and IBM on
4.2 million satellite images. We attach a UperNet decoder for binary segmentation.
Features come from layers 7, 15, 23, and 31. The decoder runs at 256 channels with
a head that steps down to 128 then 64.
"""

TRAINING_MD = """
We train for up to 80 epochs at batch size 16 on a single A100. The encoder uses a
learning rate of 5e-5, the decoder runs at 1e-4 (10x higher, since the decoder
starts from random weights). Cosine annealing, weight decay 0.1, class weights of
1.0 for background and 8.0 for fault pixels. Augmentation is horizontal flip,
vertical flip, and 90/180/270 degree rotations. Early stopping watches validation
mIoU with a patience of 15.
"""

# ---------------------------------------------------------------------------
# DIAGRAMS — drop your drawings into ./assets/ then set the src paths below.
# Until a src is set, we render a labeled placeholder box.
# ---------------------------------------------------------------------------
PIPELINE_DIAGRAM_SRC = None       # e.g. "/assets/pipeline.png"
ARCHITECTURE_DIAGRAM_SRC = None   # e.g. "/assets/architecture.png"


REFERENCES = [
    {
        "label": "Prithvi-EO-2.0 (NASA + IBM)",
        "url": "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
        "note": "Foundation model used as the backbone.",
    },
    {
        "label": "USGS Quaternary Fault and Fold Database",
        "url": "https://www.usgs.gov/programs/earthquake-hazards/faults",
        "note": "Source of fault labels.",
    },
    {
        "label": "Sentinel-2 mission",
        "url": "https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2",
        "note": "10 m multispectral imagery.",
    },
    {
        "label": "TerraTorch",
        "url": "https://github.com/IBM/terratorch",
        "note": "Framework used to fine-tune Prithvi.",
    },
    # TODO: add UperNet paper, any geomorphology references you cite
]


def _section(heading, body):
    return html.Div([
        html.H2(heading, className="mt-5 mb-3"),
        body,
    ])


def _pipeline_table(steps):
    header = html.Thead(html.Tr([
        html.Th("Step"), html.Th("Input"), html.Th("Output"),
        html.Th("Purpose"), html.Th("How / Why"),
    ]))
    rows = [
        html.Tr([
            html.Td(html.Strong(s["name"])),
            html.Td(s["input"], className="small text-muted"),
            html.Td(s["output"], className="small text-muted"),
            html.Td(s["purpose"], className="small"),
            html.Td(s["how"], className="small"),
        ])
        for s in steps
    ]
    return dbc.Table([header, html.Tbody(rows)],
                     bordered=True, hover=True, responsive=True,
                     className="align-top")


def _diagram(src, caption):
    if src:
        return html.Figure([
            html.Img(src=src, className="img-fluid"),
            html.Figcaption(caption, className="text-muted small mt-2"),
        ], className="text-center my-3")
    return html.Div([
        html.Div("Diagram goes here",
                 style={"fontSize": "1.1rem", "fontWeight": 500},
                 className="text-muted mb-2"),
        html.Div(caption, className="text-muted small"),
    ], style={
        "border": "2px dashed #ccc",
        "borderRadius": "8px",
        "padding": "4rem 1rem",
        "textAlign": "center",
        "backgroundColor": "#fafafa",
    }, className="my-3")


def _reference_item(label, url, note):
    return html.Li([
        html.A(label, href=url, target="_blank"),
        html.Span(f" — {note}", className="text-muted") if note else None,
    ])


layout = dbc.Container([

    html.Div([
        html.P("APPROACH", className="text-uppercase text-muted small mb-2"),
        html.H1("Analytical Methods", className="display-5 fw-bold mb-3"),
        html.P(
            "How the data is built and how the model is trained.",
            className="lead text-muted",
        ),
    ], className="py-4"),

    _section("Pipeline Overview", html.Div([
        dcc.Markdown(PIPELINE_OVERVIEW_MD),
        _diagram(PIPELINE_DIAGRAM_SRC,
                 "Pipeline diagram. Drop a PNG into ./assets/ and set PIPELINE_DIAGRAM_SRC."),
    ])),

    _section("Data Pipeline (Step by Step)", _pipeline_table(PIPELINE_STEPS)),

    _section("Model Architecture", html.Div([
        dcc.Markdown(MODEL_MD),
        _diagram(ARCHITECTURE_DIAGRAM_SRC,
                 "Architecture diagram. Drop a PNG into ./assets/ and set ARCHITECTURE_DIAGRAM_SRC."),
    ])),

    _section("Training Setup", dcc.Markdown(TRAINING_MD)),

    _section("References & Tools",
             html.Ul([_reference_item(**r) for r in REFERENCES])),

    html.Div(className="py-5"),

], fluid=False)
