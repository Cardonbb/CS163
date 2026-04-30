from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

# Placeholder data — replace with your dataset
df = pd.DataFrame({
    'item': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'year': [2020, 2021, 2022, 2020, 2021, 2022, 2020, 2021, 2022],
    'value': [10, 15, 20, 5, 8, 12, 30, 28, 35],
})

app = Dash()

app.layout = html.Div([
    html.H1(children='Title of Dash App', style={'textAlign': 'center'}),
    dcc.Dropdown(df.item.unique(), 'A', id='dropdown-selection'),
    dcc.Graph(id='graph-content')
])

@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_graph(value):
    dff = df[df.item == value]
    return px.line(dff, x='year', y='value')

if __name__ == '__main__':
    app.run(debug=True)
