from dash import Dash, html, dash_table, dcc
import pandas as pd
import plotly.express as px

# Placeholder data — replace with your dataset
df = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D', 'E'],
    'value1': [10, 20, 15, 25, 30],
    'value2': [5, 12, 8, 18, 22],
})

app = Dash()

app.layout = html.Div([
    html.Div(children='My First App with Data and a Graph'),
    dash_table.DataTable(data=df.to_dict('records'), page_size=10),
    dcc.Graph(figure=px.bar(df, x='category', y='value1'))
])

if __name__ == '__main__':
    app.run(debug=True)
