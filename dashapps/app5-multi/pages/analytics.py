import dash
from dash import html, dcc, callback, Input, Output

dash.register_page(__name__)

layout = html.Div([
    html.H1('This is our Analytics page'),
    html.Div([
        "Select an option: ",
        dcc.RadioItems(
            options=['Option A', 'Option B', 'Option C'],
            value='Option A',
            id='analytics-input'
        )
    ]),
    html.Br(),
    html.Div(id='analytics-output'),
])


@callback(
    Output(component_id='analytics-output', component_property='children'),
    Input(component_id='analytics-input', component_property='value')
)
def update_selected(input_value):
    return f'You selected: {input_value}'
