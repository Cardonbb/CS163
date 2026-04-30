import dash
from dash import html

dash.register_page(__name__, path='/hello', name='Hello')

layout = html.Div([
    html.H2('Hello Dash'),
    html.Div([
        html.P('Dash converts Python classes into HTML'),
        html.P("This conversion happens behind the scenes by Dash's JavaScript front-end")
    ])
])
