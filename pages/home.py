import dash
from dash import html

dash.register_page(__name__, path='/', name='Home')

layout = html.Div([
    html.H2('Home'),
    html.P('Baseline multi-page Dash site. Use the nav above to browse demos.'),
    html.Ul([
        html.Li('Basic — dropdown + line graph'),
        html.Li('Hello — HTML and styling'),
        html.Li('Table — data table + chart'),
        html.Li('Controls — radio + table + graph with a callback'),
        html.Li('Interactions — hover, click, selection events'),
        html.Li('Bootstrap Theme — themed components'),
        html.Li('Bootstrap Layout — grid layout'),
    ]),
])
