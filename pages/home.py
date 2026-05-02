import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', name='Home')

# ---------------------------------------------------------------------------
# HERO — edit the title / tagline here
# ---------------------------------------------------------------------------
HERO_TITLE = "Detecting Active Faults"
HERO_TAGLINE = "Fine-tuning a satellite foundation model to map California earthquake faults."

HEADLINE_METRICS = [
    {"label": "Fault IoU", "value": "0.55"},
    {"label": "F1 (fault)", "value": "0.71"},
    {"label": "Recall", "value": "0.71"},
    {"label": "Boundary mIoU", "value": "0.23"},
]

SUMMARY_MD = """
California has thousands of mapped active faults and likely more that have not been
found yet. We fine-tune Prithvi-EO-2.0, a 600M parameter satellite foundation model
from NASA and IBM, to flag fault traces directly in Sentinel-2 imagery. Our best run
reaches a Fault IoU of 0.55 with precision and recall both around 0.71.
"""

# ---------------------------------------------------------------------------
# Section nav cards — appear at the bottom of the home page.
# ---------------------------------------------------------------------------
SECTION_CARDS = [
    {
        "title": "Project Objectives",
        "blurb": "Goals, data sources, research questions, hypotheses.",
        "href": "/objectives",
    },
    {
        "title": "Analytical Methods",
        "blurb": "Data pipeline, model architecture, training setup.",
        "href": "/methods",
    },
    {
        "title": "Major Findings",
        "blurb": "Experiment results, threshold sweep, per-region performance.",
        "href": "/findings",
    },
]


def _metric_card(label, value):
    return dbc.Card(
        dbc.CardBody([
            html.Div(value, className="display-6 fw-bold text-primary"),
            html.Div(label, className="text-muted small"),
        ]),
        className="text-center h-100 shadow-sm",
    )


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
        html.P(HERO_TAGLINE, className="lead mb-2"),
    ], className="py-5"),

    # Headline metric strip
    dbc.Row(
        [dbc.Col(_metric_card(m["label"], m["value"]), md=3) for m in HEADLINE_METRICS],
        className="g-3 mb-5",
    ),

    # Summary block
    html.Div([
        html.H3("Project at a Glance", className="mb-3"),
        dcc.Markdown(SUMMARY_MD, className="lead"),
    ], className="mb-5"),

    # Section nav
    html.H3("Explore", className="mb-3"),
    dbc.Row(
        [dbc.Col(_section_card(**c), md=4) for c in SECTION_CARDS],
        className="g-3 mb-5",
    ),

], fluid=False)
