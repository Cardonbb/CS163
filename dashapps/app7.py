import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px

# Placeholder data — replace with your dataset
df = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D', 'E'],
    'value1': [10, 20, 15, 25, 30],
    'value2': [5, 12, 8, 18, 22],
})

app = Dash(external_stylesheets=[dbc.themes.DARKLY])

app.layout = html.Div([
    html.Div(children='My First App with Data, Graph, and Controls'),
    html.Hr(),
    dcc.RadioItems(options=['value1', 'value2'], value='value1', id='controls-and-radio-item'),
    dcc.Graph(figure={}, id='controls-and-graph')
])

@callback(
    Output(component_id='controls-and-graph', component_property='figure'),
    Input(component_id='controls-and-radio-item', component_property='value')
)
def update_graph(col_chosen):
    return px.bar(df, x='category', y=col_chosen)


if __name__ == '__main__':
    app.run(debug=True)
