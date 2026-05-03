import json
import os

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

dash.register_page(__name__, path='/', name='Home')

# ---------------------------------------------------------------------------
# Study regions — three California areas the model was trained / tested on.
# ---------------------------------------------------------------------------
STUDY_REGIONS = [
    {"name": "Bay Area",      "lat": 37.55, "lon": -122.05,
     "blurb": "Hayward & San Andreas traces"},
    {"name": "Carrizo Plain", "lat": 35.18, "lon": -119.80,
     "blurb": "Classic San Andreas exposure"},
    {"name": "Mojave",        "lat": 34.70, "lon": -116.20,
     "blurb": "Garlock & ECSZ faults"},
]

_FAULTS = {
    "San Andreas": [(-122.85, 38.15), (-122.20, 37.55), (-121.55, 36.90),
                    (-120.70, 36.10), (-119.85, 35.20), (-118.90, 34.65),
                    (-117.40, 34.10), (-116.30, 33.75), (-115.80, 33.40)],
    "Hayward":     [(-122.30, 38.05), (-122.10, 37.75), (-121.85, 37.45)],
    # Garlock — runs roughly east-west across southern California, north of Mojave.
    "Garlock":     [(-119.30, 35.20), (-118.20, 35.15), (-117.10, 35.10),
                    (-116.00, 35.10)],
}

# Real California outline (simplified) — loaded from data/california.json so
# the map does not depend on any external CDN.
_CA_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data',
                             'california.json')
with open(os.path.abspath(_CA_DATA_PATH)) as _f:
    _CA_POLYGONS = json.load(_f)['polygons']


def _study_region_map():
    fig = go.Figure()

    fig.add_shape(type="rect", xref="paper", yref="paper",
                  x0=0, x1=1, y0=0, y1=1,
                  fillcolor="#dceaf4", line_width=0, layer="below")

    for ring in _CA_POLYGONS:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            fill="toself", fillcolor="#f5efe1",
            line=dict(color="#5a5a5a", width=1.2),
            hoverinfo="skip", showlegend=False,
        ))

    for name, pts in _FAULTS.items():
        xs, ys = zip(*pts)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="#8b1a1a", width=2, dash="dash"),
            hovertemplate=f"{name} fault<extra></extra>",
            showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=[r["lon"] for r in STUDY_REGIONS],
        y=[r["lat"] for r in STUDY_REGIONS],
        mode="markers+text",
        marker=dict(size=16, color="#d62728",
                    line=dict(width=2.5, color="white")),
        text=[r["name"] for r in STUDY_REGIONS],
        # Per-marker offset so the Mojave label does not crowd Carrizo Plain.
        textposition=["middle right", "middle left", "middle right"],
        textfont=dict(size=13, color="#1a1a1a"),
        hovertext=[f"<b>{r['name']}</b><br>{r['blurb']}" for r in STUDY_REGIONS],
        hoverinfo="text",
        showlegend=False,
        cliponaxis=False,
    ))

    fig.update_xaxes(range=[-125.0, -113.5], visible=False)
    fig.update_yaxes(range=[32.0, 42.5], visible=False,
                     scaleanchor="x", scaleratio=1.25)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        plot_bgcolor="#dceaf4",
        paper_bgcolor="#dceaf4",
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    return fig


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------
HERO_TITLE = "Detecting Active Faults"
HERO_TAGLINE = ("Fine-tuning a satellite foundation model to map "
                "California earthquake faults.")

SUMMARY_MD = """
California has thousands of mapped active faults and likely more that have not
been found yet. This project fine-tunes **Prithvi-EO-2.0**, a 600M-parameter
satellite foundation model from NASA and IBM, to flag fault traces directly in
Sentinel-2 imagery across three study regions: the Bay Area, Carrizo Plain, and
the Mojave Desert.
"""

GOAL_MD = """
Build a model that looks at Sentinel-2 satellite images and predicts which
pixels are likely part of an active fault. The goal is not to replace
geologists, but to give them a better starting point when deciding where to
investigate. California has over 15,000 km of mapped active faults, and the
2019 Ridgecrest earthquake showed that some important faults are still not
fully mapped. A faster screening tool could help flag more possible fault
areas before they become obvious through a major earthquake.
"""

RESEARCH_QUESTIONS = [
    "Can a pretrained satellite foundation model learn fault signatures from a small labeled dataset?",
    "Which augmentation matters most when fault traces run at every angle?",
    "How well does a model trained on three California regions generalize to terrain it has never seen?",
]

HYPOTHESES = [
    "Fine-tuning Prithvi-EO-2.0 on our ~4,200 labeled patches should help the model learn real fault patterns, not just guess from the raw satellite image.",
    "Rotation augmentation should help more than flips alone because faults can run in many different directions.",
    "Performance will likely differ by region. Clearer terrain, such as Carrizo Plain, should be easier for the model, while more visually noisy areas, such as the Mojave Desert, may create more false positives.",
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
