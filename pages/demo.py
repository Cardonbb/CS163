"""Interactive demo — calls the Cloud Run inference service."""

import os

import dash
import dash_bootstrap_components as dbc
import requests
from dash import Input, Output, State, callback, dcc, html

dash.register_page(__name__, path="/demo", name="Live Demo")

INFERENCE_URL = os.environ.get("INFERENCE_URL", "").rstrip("/")


def _fetch_patch_list() -> list[str]:
    if not INFERENCE_URL:
        return []
    try:
        resp = requests.get(f"{INFERENCE_URL}/patches", timeout=10)
        return resp.json().get("patches", [])
    except Exception:
        return []


_PATCHES = _fetch_patch_list()

_PANEL_STYLE = {"textAlign": "center", "fontWeight": "600",
                "marginBottom": "6px", "fontSize": "0.9rem"}

layout = dbc.Container([

    html.Div([
        html.P("LIVE DEMO", className="text-uppercase text-muted small mb-2"),
        html.H1("Fault Detection Demo", className="display-5 fw-bold mb-3"),
        html.P(
            "Pick any test patch to see the satellite image, the USGS ground "
            "truth, and your model's prediction side by side.",
            className="lead text-muted",
        ),
    ], className="py-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Test patch", className="fw-semibold"),
            dcc.Dropdown(
                id="demo-patch",
                options=[{"label": p, "value": p} for p in _PATCHES],
                value=_PATCHES[0] if _PATCHES else None,
                placeholder="Deploy the service to populate this list",
                clearable=False,
            ),
        ], md=7),
        dbc.Col([
            html.Br(),
            dbc.Button("Run Inference", id="demo-run-btn", color="primary",
                       className="mt-1", disabled=not bool(_PATCHES)),
        ], md=3),
    ], className="mb-4"),

    dcc.Loading(html.Div(id="demo-output"), type="circle"),

    html.Div(className="py-5"),

], fluid=False)


def _img_card(title: str, b64: str) -> dbc.Col:
    return dbc.Col(
        html.Div([
            html.Div(title, style=_PANEL_STYLE),
            html.Img(
                src=f"data:image/png;base64,{b64}",
                style={"width": "100%", "imageRendering": "pixelated",
                       "borderRadius": "4px"},
            ),
        ]),
        md=4,
    )


@callback(
    Output("demo-output", "children"),
    Input("demo-run-btn", "n_clicks"),
    State("demo-patch", "value"),
    prevent_initial_call=True,
)
def run_inference(n_clicks, patch):
    if not INFERENCE_URL:
        return dbc.Alert(
            [html.Strong("INFERENCE_URL not configured. "),
             "Add it to app.yaml and redeploy."],
            color="warning",
        )

    if not patch:
        return dbc.Alert("Select a patch first.", color="secondary")

    try:
        resp = requests.get(
            f"{INFERENCE_URL}/predict",
            params={"patch": patch},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as exc:
        msg = exc.response.json().get("error", str(exc)) if exc.response else str(exc)
        return dbc.Alert(f"Service error: {msg}", color="danger")
    except Exception as exc:
        return dbc.Alert(f"Could not reach inference service: {exc}", color="danger")

    fault_pct = data["fault_pixel_fraction"] * 100
    gt_pct    = data["ground_truth_fraction"] * 100

    return [
        dbc.Row([
            _img_card("Sentinel-2 RGB",   data["rgb_png"]),
            _img_card("Ground Truth",      data["ground_truth_png"]),
            _img_card("Model Prediction",  data["prediction_png"]),
        ], className="g-3 mb-3"),
        dbc.Alert(
            [html.Strong(patch), f" — model flagged {fault_pct:.1f}% of pixels as fault "
             f"(ground truth: {gt_pct:.1f}%)."],
            color="info",
        ),
    ]
