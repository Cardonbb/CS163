import dash
from dash import html, dcc, callback, Output, Input
import plotly.express as px
import json
import pandas as pd

dash.register_page(__name__, path='/interactions', name='Interactions')

df = pd.DataFrame({
    "x": [1, 2, 1, 2],
    "y": [1, 2, 3, 4],
    "customdata": [1, 2, 3, 4],
    "fruit": ["apple", "apple", "orange", "orange"]
})

fig = px.scatter(df, x="x", y="y", color="fruit", custom_data=["customdata"])
fig.update_layout(clickmode='event+select')
fig.update_traces(marker_size=20)

pre_style = {'border': 'thin lightgrey solid', 'overflowX': 'scroll', 'padding': '6px'}

layout = html.Div([
    html.H2('Interactions — Hover, Click, Selection'),
    dcc.Graph(id='interactions-graph', figure=fig),
    html.Div([
        html.Div([
            dcc.Markdown('**Hover Data**'),
            html.Pre(id='hover-data', style=pre_style),
        ], style={'flex': 1, 'padding': '6px'}),
        html.Div([
            dcc.Markdown('**Click Data**'),
            html.Pre(id='click-data', style=pre_style),
        ], style={'flex': 1, 'padding': '6px'}),
        html.Div([
            dcc.Markdown('**Selection Data**'),
            html.Pre(id='selected-data', style=pre_style),
        ], style={'flex': 1, 'padding': '6px'}),
        html.Div([
            dcc.Markdown('**Relayout Data**'),
            html.Pre(id='relayout-data', style=pre_style),
        ], style={'flex': 1, 'padding': '6px'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),
])


@callback(Output('hover-data', 'children'), Input('interactions-graph', 'hoverData'))
def display_hover(hoverData):
    return json.dumps(hoverData, indent=2)


@callback(Output('click-data', 'children'), Input('interactions-graph', 'clickData'))
def display_click(clickData):
    return json.dumps(clickData, indent=2)


@callback(Output('selected-data', 'children'), Input('interactions-graph', 'selectedData'))
def display_selected(selectedData):
    return json.dumps(selectedData, indent=2)


@callback(Output('relayout-data', 'children'), Input('interactions-graph', 'relayoutData'))
def display_relayout(relayoutData):
    return json.dumps(relayoutData, indent=2)
