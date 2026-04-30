import dash
from dash import html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px

dash.register_page(__name__, path='/controls', name='Controls')

# Placeholder data — replace with your dataset
df = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D', 'E'],
    'value1': [10, 20, 15, 25, 30],
    'value2': [5, 12, 8, 18, 22],
})

layout = html.Div([
    html.H2('Controls — Radio + Table + Graph'),
    html.Hr(),
    dcc.RadioItems(options=['value1', 'value2'], value='value1', id='controls-radio'),
    dash_table.DataTable(data=df.to_dict('records'), page_size=6),
    dcc.Graph(figure={}, id='controls-graph'),
])

@callback(
    Output('controls-graph', 'figure'),
    Input('controls-radio', 'value')
)
def update_graph(col_chosen):
    return px.bar(df, x='category', y=col_chosen)
