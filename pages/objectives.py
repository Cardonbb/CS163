import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/objectives', name='Objectives')


PROJECT_GOAL_MD = """
TODO
"""

BROADER_IMPACT_MD = """
TODO
"""

RESEARCH_QUESTIONS = [
    "TODO",
]

HYPOTHESES = [
    "TODO",
]

DATA_SOURCES = [
    {
        "name": "Sentinel-2",
        "description": "10 m multispectral imagery (RGB + NIR + SWIR1/2). "
                       "Median composite per region; SCL cloud mask; "
                       "dry-season acquisition for rock/soil contrast.",
        "url": "https://eos.com/find-satellite/sentinel-2/",
    },
    {
        "name": "NAIP",
        "description": "1 m resolution aerial imagery. Planned for high-resolution "
                       "validation passes — TODO confirm scope.",
        "url": "https://naip-usdaonline.hub.arcgis.com/",
    },
    {
        "name": "USGS Quaternary Fault and Fold Database",
        "description": "Vector fault traces filtered to active California "
                       "segments; buffered 50 m and rasterized into binary masks.",
        "url": "https://www.usgs.gov/programs/earthquake-hazards/faults",
    },
    # TODO: add LiDAR DEM source if/when integrated
]

OUTCOMES_MD = """

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
