import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/bootstrap-layout', name='Bootstrap Layout')

layout = dbc.Container([
    html.H2('Bootstrap Layout — Grid'),
    html.Hr(),
    dbc.Row(dbc.Col(html.Div("A single column"))),
    html.Br(),
    dbc.Row([
        dbc.Col(html.Div("One of three columns")),
        dbc.Col(html.Div("One of three columns")),
        dbc.Col(html.Div("One of three columns")),
    ]),
])
