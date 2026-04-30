import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

nav = html.Div(
    [
        dcc.Link(
            f"{page['name']}",
            href=page["relative_path"],
            style={'marginRight': '15px'}
        )
        for page in dash.page_registry.values()
    ],
    style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'marginBottom': '20px'}
)

app.layout = html.Div([
    html.H1('My Dash Site', style={'textAlign': 'center', 'padding': '10px'}),
    nav,
    html.Div(dash.page_container, style={'padding': '20px'})
])


if __name__ == '__main__':
    app.run(debug=True)
