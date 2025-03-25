"""
newsies.api.dash_app
"""

from fastapi.middleware.wsgi import WSGIMiddleware

dash_app = Dash(__name__)


def fetch_graph_data():
    """Fetch graph data from FastAPI"""
    response = requests.get("http://127.0.0.1:8000/graph_data")
    return response.json()


# Dash Layout
dash_app.layout = html.Div(
    [
        html.H1("Article Clustering"),
        cyto.Cytoscape(
            id="article-graph",
            layout={"name": "cose"},
            style={"width": "100%", "height": "600px"},
            elements=fetch_graph_data(),
            stylesheet=[
                {"selector": "node", "style": {"label": "data(label)"}},
                {"selector": "edge", "style": {"width": 2, "line-color": "#888"}},
            ],
        ),
    ]
)

# Mount Dash to FastAPI
app.mount("/dashboard", WSGIMiddleware(dash_app.server))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
