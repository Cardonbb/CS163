import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/methods', name='Methods')

# ---------------------------------------------------------------------------
# This page can be more technical than the others. Each PIPELINE_STEPS entry
# is one row in the methods table — the same shape as your Analysis Method doc.
# ---------------------------------------------------------------------------

PIPELINE_OVERVIEW_MD = """
TODO
"""

PIPELINE_STEPS = [
    {
        "name": "TODO",
        "input": "TODO",
        "output": "TODO",
        "purpose": "TODO",
        "how": "TODO",
    },
]

MODEL_MD = """
TODO
"""

TRAINING_MD = """
TODO
"""

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

    _section("Pipeline Overview", dcc.Markdown(PIPELINE_OVERVIEW_MD)),

    _section("Data Pipeline (Step by Step)", _pipeline_table(PIPELINE_STEPS)),

    _section("Model Architecture", dcc.Markdown(MODEL_MD)),

    _section("Training Setup", dcc.Markdown(TRAINING_MD)),

    _section("References & Tools",
             html.Ul([_reference_item(**r) for r in REFERENCES])),

    html.Div(className="py-5"),

], fluid=False)
