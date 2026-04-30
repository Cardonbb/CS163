import dash
from dash import html, dash_table, dcc
import pandas as pd
import plotly.express as px

dash.register_page(__name__, path='/table', name='Table')

# Placeholder data — replace with your dataset
df = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D', 'E'],
    'value1': [10, 20, 15, 25, 30],
    'value2': [5, 12, 8, 18, 22],
})

layout = html.Div([
    html.H2('Table — Data Table + Chart'),
    dash_table.DataTable(data=df.to_dict('records'), page_size=10),
    dcc.Graph(figure=px.bar(df, x='category', y='value1'))
])
