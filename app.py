import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

# Sample data
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Grapes", "Strawberries"],
    "Amount": [4, 1, 2, 5, 3],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal"],
})

app.layout = html.Div(
    className="app-container",
    children=[
        html.H1("Dash App Template", className="app-title"),

        html.Div(
            className="controls",
            children=[
                html.Label("Select City:"),
                dcc.Dropdown(
                    id="city-dropdown",
                    options=[{"label": c, "value": c} for c in df["City"].unique()],
                    value=df["City"].iloc[0],
                    clearable=False,
                ),
            ],
        ),

        dcc.Graph(id="bar-chart"),

        html.Div(id="summary-text", className="summary"),
    ],
)


@app.callback(
    Output("bar-chart", "figure"),
    Output("summary-text", "children"),
    Input("city-dropdown", "value"),
)
def update_chart(selected_city):
    filtered = df[df["City"] == selected_city]
    fig = px.bar(filtered, x="Fruit", y="Amount", title=f"Fruit Amounts in {selected_city}")
    summary = f"Showing {len(filtered)} items for {selected_city}."
    return fig, summary


if __name__ == "__main__":
    # Set debug=False (or remove it) before deploying to production
    app.run(debug=True)
