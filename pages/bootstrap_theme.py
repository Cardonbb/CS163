import dash
from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

dash.register_page(__name__, path='/bootstrap-theme', name='Bootstrap Theme')

# Placeholder data — replace with your dataset
df = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D', 'E'],
    'value1': [10, 20, 15, 25, 30],
    'value2': [5, 12, 8, 18, 22],
})

layout = dbc.Container([
    html.H2('Bootstrap Theme — Themed Components'),
    html.Hr(),
    dbc.Card([
        dbc.CardBody([
            dcc.RadioItems(
                options=['value1', 'value2'],
                value='value1',
                id='bs-theme-radio'
            ),
            dcc.Graph(figure={}, id='bs-theme-graph'),
        ])
    ]),
])


@callback(
    Output('bs-theme-graph', 'figure'),
    Input('bs-theme-radio', 'value')
)
def update_graph(col_chosen):
    return px.bar(df, x='category', y=col_chosen)
