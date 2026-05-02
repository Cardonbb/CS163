import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/objectives', name='Objectives')


PROJECT_GOAL_MD = """
Build a model that takes a Sentinel-2 image and returns a per-pixel prediction of
whether each pixel sits on an active fault. The point is to help geologists narrow
down where to look in the field, not replace them.
"""

BROADER_IMPACT_MD = """
California has more than 15,000 km of mapped active faults, and the 2019 Ridgecrest
earthquake happened on a fault that was not fully mapped beforehand. A faster
screening tool means more candidate faults get a second look before the next
earthquake reveals them.
"""

RESEARCH_QUESTIONS = [
    "Can a pretrained satellite foundation model learn fault signatures from a small labeled dataset?",
    "Which augmentation matters most when fault traces run at every angle?",
    "How well does a model trained on three California regions generalize to terrain it has never seen?",
]

HYPOTHESES = [
    "Rotation augmentation will outperform flips alone since faults have no preferred orientation.",
    "Class-weighted cross entropy is enough to handle the 1.2 percent fault pixel imbalance.",
    "Boundary mIoU will improve more than pixel accuracy as the model learns thin linear traces.",
]

DATA_SOURCES = [
    {
        "name": "Sentinel-2",
        "description": "TODO",
        "url": "https://eos.com/find-satellite/sentinel-2/",
    },
    {
        "name": "NAIP",
        "description": "TODO",
        "url": "https://naip-usdaonline.hub.arcgis.com/",
    },
    {
        "name": "USGS Quaternary Fault and Fold Database",
        "description": "TODO",
        "url": "https://www.usgs.gov/programs/earthquake-hazards/faults",
    },
    # TODO: add LiDAR DEM source if/when integrated
]

OUTCOMES_MD = """
A working segmentation model with a published threshold sweep, plus per-region IoU
so users can see where the predictions hold up and where they do not.
"""


def _section(heading, body):
    return html.Div([
        html.H2(heading, className="mt-5 mb-3"),
        body,
    ])


def _data_source_card(name, description, url):
    return dbc.Card(
        dbc.CardBody([
            html.H5(name, className="card-title"),
            html.P(description, className="text-muted small"),
            html.A("Source ↗", href=url, target="_blank") if url else None,
        ]),
        className="h-100 shadow-sm",
    )


layout = dbc.Container([

    # Page header
    html.Div([
        html.P("PROJECT", className="text-uppercase text-muted small mb-2"),
        html.H1("Objectives", className="display-5 fw-bold mb-3"),
        html.P(
            "What we're building, why it matters, and how we'll know if it worked.",
            className="lead text-muted",
        ),
    ], className="py-4"),

    _section("Goal", dcc.Markdown(PROJECT_GOAL_MD)),

    _section("Broader Impact", dcc.Markdown(BROADER_IMPACT_MD)),

    _section("Research Questions",
             html.Ol([html.Li(q) for q in RESEARCH_QUESTIONS])),

    _section("Hypotheses",
             html.Ul([html.Li(h) for h in HYPOTHESES])),

    _section("Data Sources",
             dbc.Row(
                 [dbc.Col(_data_source_card(**ds), md=6, className="mb-3")
                  for ds in DATA_SOURCES],
                 className="g-3",
             )),

    _section("Expected Outcomes", dcc.Markdown(OUTCOMES_MD)),

    html.Div(className="py-5"),

], fluid=False)
