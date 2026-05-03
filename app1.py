import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Detecting Active Faults",
)
server = app.server

# Explicit page order so the navbar matches the rubric flow.
NAV_ORDER = ["/", "/eda", "/methods", "/findings"]

def _ordered_pages():
    by_path = {p["relative_path"]: p for p in dash.page_registry.values()}
    ordered = [by_path[path] for path in NAV_ORDER if path in by_path]
    extras = [p for p in dash.page_registry.values()
              if p["relative_path"] not in NAV_ORDER]
    return ordered + extras

navbar = dbc.NavbarSimple(
    brand="Detecting Active Faults",
    brand_href="/",
    color="dark",
    dark=True,
    fluid=True,
    children=[
        dbc.NavItem(dcc.Link(p["name"], href=p["relative_path"],
                             className="nav-link"))
        for p in _ordered_pages()
    ],
)

app.layout = html.Div([
    navbar,
    html.Div(dash.page_container),
])


if __name__ == '__main__':
    app.run(debug=True)
