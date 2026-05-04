import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from cloud_io import load_json

dash.register_page(__name__, path='/', name='Home')

# ---------------------------------------------------------------------------
# Study regions — three California areas the model was trained / tested on.
# ---------------------------------------------------------------------------
STUDY_REGIONS = [
    {"name": "Bay Area",      "lat": 37.55, "lon": -122.05,
     "blurb": "Hayward & Calaveras faults — urban/oak woodland, hardest environment"},
    {"name": "Carrizo Plain", "lat": 35.18, "lon": -119.80,
     "blurb": "Central San Andreas — textbook fault geomorphology"},
    {"name": "Mojave",        "lat": 34.70, "lon": -116.20,
     "blurb": "S. San Andreas & Mojave faults — arid desert, scarps highly visible"},
]

_FAULTS = {
    "San Andreas": [(-122.85, 38.15), (-122.20, 37.55), (-121.55, 36.90),
                    (-120.70, 36.10), (-119.85, 35.20), (-118.90, 34.65),
                    (-117.40, 34.10), (-116.30, 33.75), (-115.80, 33.40)],
    "Hayward":     [(-122.30, 38.05), (-122.10, 37.75), (-121.85, 37.45)],
    "Calaveras":   [(-121.80, 37.60), (-121.60, 37.20), (-121.40, 36.80)],
    "Garlock":     [(-119.30, 35.20), (-118.20, 35.15), (-117.10, 35.10),
                    (-116.00, 35.10)],
}

# Study region bounding boxes from report Table 1
_REGION_BOXES = [
    {"name": "Bay Area",      "x0": -122.5, "x1": -121.5, "y0": 37.5, "y1": 38.2, "color": "#1f77b4"},
    {"name": "Carrizo Plain", "x0": -120.2, "x1": -119.5, "y0": 35.0, "y1": 35.6, "color": "#2ca02c"},
    {"name": "Mojave",        "x0": -116.5, "x1": -115.5, "y0": 33.8, "y1": 34.5, "color": "#ff7f0e"},
]

# Real California outline (simplified). At runtime we try
# gs://cs163-fault-data-carter/site/california.json and fall back to the
# bundled copy under data/ so the site still works offline.
_CA_POLYGONS = load_json("site/california.json", "data/california.json")["polygons"]


_FAULT_COLORS = {
    "San Andreas": "#8b1a1a",
    "Hayward":     "#c0392b",
    "Calaveras":   "#e67e22",
    "Garlock":     "#6c3483",
}


def _study_region_map():
    fig = go.Figure()

    # Ocean background
    fig.add_shape(type="rect", xref="paper", yref="paper",
                  x0=0, x1=1, y0=0, y1=1,
                  fillcolor="#dceaf4", line_width=0, layer="below")

    # California outline
    for ring in _CA_POLYGONS:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            fill="toself", fillcolor="#f5efe1",
            line=dict(color="#5a5a5a", width=1.2),
            hoverinfo="skip", showlegend=False,
        ))

    # Study region markers
    for i, r in enumerate(STUDY_REGIONS):
        box = _REGION_BOXES[i]
        fig.add_trace(go.Scatter(
            x=[r["lon"]], y=[r["lat"]],
            mode="markers+text",
            name=r["name"],
            marker=dict(size=18, color=box["color"],
                        symbol="star", line=dict(width=1.5, color="white")),
            text=[r["name"]],
            textposition=["middle right", "middle left", "middle right"][i],
            textfont=dict(size=13, color="#1a1a1a", family="Arial Black"),
            hovertext=f"<b>{r['name']}</b><br>{r['blurb']}",
            hoverinfo="text",
            showlegend=False,
        ))

    fig.update_xaxes(range=[-125.0, -113.5], visible=False)
    fig.update_yaxes(range=[32.0, 42.5], visible=False,
                     scaleanchor="x", scaleratio=1.25)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        plot_bgcolor="#dceaf4",
        paper_bgcolor="#dceaf4",
        hoverlabel=dict(bgcolor="white", font_size=13),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1,
            font=dict(size=12),
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------
HERO_TITLE = "FaultFinder"
HERO_TAGLINE = ("Fine-tuning a geospatial foundation model to detect active "
                "earthquake faults in Sentinel-2 satellite imagery.")

SUMMARY_MD = """
FaultFinder fine-tunes **Prithvi-EO 2.0**, a 600M-parameter Vision Transformer
pretrained by IBM and NASA, to perform binary segmentation of active fault traces
in Sentinel-2 imagery. We built a data pipeline and generated a dataset of 4,207 paired 128×128
image-mask patches from three environmentally diverse California regions. The best model
achieves Fault IoU 0.5523, F1 0.7116, and mIoU 0.7473. This project relies on optical imagery and 
focusing on using 6 bands of Sentinel-2 imagery to detect active fault traces.
"""

GOAL_MD = """
Earthquakes are one of the most costly natural hazards globally. In California alone, we 
can see most earthquakes are in the San Francisco Bay Area. Accurate seismic
hazard models depend critically on fault geometry, a fault not on the map cannot
be assessed for risk. The 2019 Mw 7.1 Ridgecrest earthquake ruptured a fault
network where only 35% of surface rupture traces had been previously mapped,
causing ~$4 billion in damage. Geologist have many unmapped active faults and inaccurate mappings as well. 
If they can find more active faults, they can assess seismic hazard more accurately. The goal is not to replace
geologists, but to give them a faster starting point for identifying candidate faults
before a major earthquake makes them obvious.
"""

RESEARCH_QUESTIONS = [
    "Can a pretrained geospatial foundation model learn fault signatures from a small labeled dataset of ~4,200 patches?",
    "Which augmentation strategy matters most when fault traces run at every orientation?",
    "Does Dice loss improve or degrade performance on buffered linear fault labels?",
]

HYPOTHESES = [
    "Fine-tuning Prithvi-EO 2.0 on labeled patches will produce fault representations that are more accurate and reliable.",
    "Rotation augmentation will outperform flips alone because geological faults have no preferred orientation and Prithvi's positional encodings hard-code directional bias from north-up pretraining.",
    "Dice loss will underperform weighted cross-entropy due to its region-size bias against the 50-meter buffered linear labels used as ground truth.",
]

SECTION_CARDS = [
    {"title": "Exploratory Data Analysis",
     "blurb": "Dataset summary, region splits, sample patches, distributions.",
     "href": "/eda"},
    {"title": "Analysis Methods",
     "blurb": "Data pipeline, model architecture, training setup.",
     "href": "/methods"},
    {"title": "Major Findings",
     "blurb": "Experiment results, threshold sweep, per-region performance.",
     "href": "/findings"},
]


def _section_card(title, blurb, href):
    return dbc.Card(
        dbc.CardBody([
            html.H5(title, className="card-title"),
            html.P(blurb, className="card-text text-muted"),
            dcc.Link("Read more →", href=href, className="stretched-link"),
        ]),
        className="h-100 shadow-sm",
    )


layout = dbc.Container([

    # Hero
    html.Div([
        html.H1(HERO_TITLE, className="display-4 fw-bold mb-3"),
        html.P(HERO_TAGLINE, className="lead mb-0"),
    ], className="py-5"),

    # Project summary
    html.Div([
        html.H3("Project Summary", className="mb-3"),
        dcc.Markdown(SUMMARY_MD),
    ], className="mb-5"),

    # Project goals + research questions + hypotheses
    html.Div([
        html.H3("Project Goals", className="mb-3"),
        dcc.Markdown(GOAL_MD),
        dbc.Row([
            dbc.Col([
                html.H5("Research Questions", className="mt-3"),
                html.Ol([html.Li(q) for q in RESEARCH_QUESTIONS]),
            ], md=6),
            dbc.Col([
                html.H5("Hypotheses", className="mt-3"),
                html.Ul([html.Li(h) for h in HYPOTHESES]),
            ], md=6),
        ], className="g-3"),
    ], className="mb-5"),

    # Study-region map
    html.Div([
        html.H3("Study Regions", className="mb-3"),
        dcc.Graph(figure=_study_region_map(),
                  config={"displayModeBar": False}),
        html.P(
            "Three California regions — Bay Area, Carrizo Plain, and Mojave — "
            "covering roughly 300 km² of labeled fault terrain. Hover a marker "
            "for region notes.",
            className="text-muted small text-center mt-2",
        ),
    ], className="mb-5"),

    # Explore links
    html.H3("Explore", className="mb-3"),
    dbc.Row(
        [dbc.Col(_section_card(**c), md=4) for c in SECTION_CARDS],
        className="g-3 mb-5",
    ),

], fluid=False)
