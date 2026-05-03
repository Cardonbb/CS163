from collections import Counter

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import cloud_io
from cloud_io import gcs_uri, load_json, load_text

dash.register_page(__name__, path='/eda', name='EDA')

# ---------------------------------------------------------------------------
# Dataset facts loaded from GCS at startup, with a local fallback.
#   gs://cs163-fault-data-carter/dataset/dataset_info.json
#   gs://cs163-fault-data-carter/dataset/splits/{train,val,test}.txt
# ---------------------------------------------------------------------------
DATASET_INFO = load_json(
    "dataset/dataset_info.json",
    "contextOfProject/patches/dataset_info.json",
)


def _split_counts():
    """Count patches per region per split, reading the split lists from GCS."""
    counts = {}
    for split in ("train", "val", "test"):
        names = load_text(
            f"dataset/splits/{split}.txt",
            f"contextOfProject/patches/splits/{split}.txt",
        ).split()
        counts[split] = dict(Counter(n.split("_r")[0] for n in names))
    return counts


REGION_SPLIT_COUNTS = _split_counts()
# Snapshot which source actually served the data (gcs / local) — used in the UI.
DATA_SOURCE = cloud_io.GCS_SOURCE

REGIONS = ["bay_area", "carrizo", "mojave"]
SPLIT_COLORS = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}

SUMMARY_BULLETS = [
    "4,207 image / label patches at 128×128 pixels, drawn from three California regions.",
    "Sentinel-2 imagery, 6 surface-reflectance bands (Blue, Green, Red, NIR, SWIR1, SWIR2) at 10 m resolution.",
    "Pixel labels are USGS Quaternary Fault and Fold traces, buffered 50 m on each side and rasterized.",
    "Patches were chipped with a sliding window at stride 64 and filtered to keep only those with at least 0.5% fault pixels.",
    "Split into train (2,944) / val (631) / test (632) by spatial chunks so nearby patches do not leak across splits.",
]

PREVIEW_FILES = [
    "bay_area_r03904_c02368.npy",
    "carrizo_r04480_c04160.npy",
    "mojave_r05888_c01216.npy",
    "mojave_r07936_c02112.npy",
]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _split_pie():
    labels = ["Train", "Val", "Test"]
    values = [DATASET_INFO[f"{k.lower()}_patches"] for k in ["train", "val", "test"]]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.45,
        marker=dict(colors=[SPLIT_COLORS["train"], SPLIT_COLORS["val"],
                            SPLIT_COLORS["test"]]),
        textinfo="label+value+percent",
    ))
    fig.update_layout(
        title="Train / Val / Test Split",
        height=380, margin=dict(l=20, r=20, t=50, b=20), showlegend=False,
    )
    return fig


def _region_split_bar():
    fig = go.Figure()
    for split in ["train", "val", "test"]:
        fig.add_trace(go.Bar(
            name=split.title(),
            x=[r.replace("_", " ").title() for r in REGIONS],
            y=[REGION_SPLIT_COUNTS[split][r] for r in REGIONS],
            marker_color=SPLIT_COLORS[split],
        ))
    fig.update_layout(
        barmode="stack",
        title="Patches per Region (stacked by split)",
        xaxis_title="Region", yaxis_title="Patch count",
        height=380, margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return fig


def _region_share_bar():
    totals = {r: sum(REGION_SPLIT_COUNTS[s][r] for s in REGION_SPLIT_COUNTS)
              for r in REGIONS}
    grand = sum(totals.values())
    fig = go.Figure(go.Bar(
        x=[r.replace("_", " ").title() for r in REGIONS],
        y=[100 * totals[r] / grand for r in REGIONS],
        marker_color="#7d3c98",
        text=[f"{100*totals[r]/grand:.1f}%" for r in REGIONS],
        textposition="outside",
    ))
    fig.update_layout(
        title="Region share of total patches",
        yaxis_title="Share (%)", yaxis_range=[0, 65],
        height=380, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _bands_bar():
    fig = go.Figure(go.Bar(
        x=DATASET_INFO["bands"],
        y=[1] * len(DATASET_INFO["bands"]),
        marker_color=["#3498db", "#2ecc71", "#e74c3c",
                      "#9b59b6", "#f39c12", "#34495e"],
        hovertemplate="Band: %{x}<extra></extra>",
    ))
    fig.update_layout(
        title="Sentinel-2 bands used (10 m, surface reflectance)",
        yaxis=dict(visible=False), xaxis_title="",
        height=300, margin=dict(l=20, r=20, t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(heading, *body):
    return html.Div([
        html.H2(heading, className="mt-5 mb-3"),
        *body,
    ])


def _stats_table():
    rows = [
        ("Total patches",    f"{DATASET_INFO['total_patches']:,}"),
        ("Train",            f"{DATASET_INFO['train_patches']:,}"),
        ("Val",              f"{DATASET_INFO['val_patches']:,}"),
        ("Test",             f"{DATASET_INFO['test_patches']:,}"),
        ("Patch size",       f"{DATASET_INFO['patch_size']} × {DATASET_INFO['patch_size']} px"),
        ("Stride",           f"{DATASET_INFO['stride']} px"),
        ("Min fault fraction (per patch)", f"{100*DATASET_INFO['min_fault_frac']:.1f}%"),
        ("Number of bands",  str(DATASET_INFO["num_bands"])),
        ("Regions",          ", ".join(r.replace("_", " ").title()
                                       for r in DATASET_INFO["regions"])),
    ]
    return dbc.Table(
        [html.Thead(html.Tr([html.Th("Field"), html.Th("Value")]))] +
        [html.Tbody([html.Tr([html.Td(k), html.Td(v)]) for k, v in rows])],
        bordered=True, striped=True, hover=True, size="sm",
    )


def _preview_table():
    rows = []
    for fname in PREVIEW_FILES:
        region, rest = fname.split("_r", 1)
        row, col = rest.replace(".npy", "").split("_c")
        rows.append((fname, region.replace("_", " ").title(), row, col))
    return dbc.Table(
        [html.Thead(html.Tr([html.Th("File"), html.Th("Region"),
                             html.Th("Row offset"), html.Th("Col offset")]))] +
        [html.Tbody([html.Tr([html.Td(f), html.Td(r), html.Td(rw), html.Td(c)])
                     for f, r, rw, c in rows])],
        bordered=True, striped=True, hover=True, size="sm",
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = dbc.Container([

    html.Div([
        html.P("DATA", className="text-uppercase text-muted small mb-2"),
        html.H1("Exploratory Data Analysis", className="display-5 fw-bold mb-3"),
        html.P(
            "What the patches look like, how they are split across regions, "
            "and what we did to keep the labels usable.",
            className="lead text-muted",
        ),
        html.Div([
            dbc.Badge(
                f"Data source: {DATA_SOURCE.upper()}",
                color="success" if DATA_SOURCE == "gcs" else "secondary",
                className="me-2",
            ),
            html.Code(gcs_uri("dataset/")),
        ], className="small mb-2"),
    ], className="py-4"),

    _section("Dataset Summary",
             html.Ul([html.Li(b) for b in SUMMARY_BULLETS])),

    _section("Dataset Statistics", _stats_table()),

    _section(
        "Sample Patch Filenames",
        html.P(
            "Each patch is stored as a 6-band float32 .npy under "
            "patches/images/, with a matching binary label under "
            "patches/labels/. Filenames encode the region and the "
            "(row, col) pixel offset of the patch within that region's "
            "source mosaic.",
            className="text-muted small",
        ),
        _preview_table(),
    ),

    _section(
        "Distributions",
        dbc.Row([
            dbc.Col(dcc.Graph(figure=_split_pie(),
                              config={"displayModeBar": False}), md=6),
            dbc.Col(dcc.Graph(figure=_region_share_bar(),
                              config={"displayModeBar": False}), md=6),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=_region_split_bar(),
                              config={"displayModeBar": False}), md=6),
            dbc.Col(dcc.Graph(figure=_bands_bar(),
                              config={"displayModeBar": False}), md=6),
        ], className="g-3 mt-2"),
    ),

    _section(
        "Sample Predictions on Held-out Test Patches",
        html.P(
            "Six randomly chosen test tiles. Top: Sentinel-2 RGB input. "
            "Middle: USGS ground-truth fault traces (50 m buffer). "
            "Bottom: model prediction at threshold 0.65.",
            className="text-muted small",
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
                     style={"flex": "1 1 auto", "minWidth": 0}),
        ], className="d-flex align-items-stretch"),
    ),

    html.Div(className="py-5"),

], fluid=False)
