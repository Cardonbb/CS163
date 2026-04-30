import dash
from dash import html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

dash.register_page(__name__, path='/basic', name='Basic')

# Placeholder data — replace with your dataset
df = pd.DataFrame({
    'item': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'year': [2020, 2021, 2022, 2020, 2021, 2022, 2020, 2021, 2022],
    'value': [10, 15, 20, 5, 8, 12, 30, 28, 35],
})

layout = html.Div([
    html.H2('Basic — Dropdown + Line'),
    dcc.Dropdown(df.item.unique(), 'A', id='basic-dropdown'),
    dcc.Graph(id='basic-graph'),
])

@callback(
    Output('basic-graph', 'figure'),
    Input('basic-dropdown', 'value')
)
def update_graph(value):
    dff = df[df.item == value]
    return px.line(dff, x='year', y='value')
