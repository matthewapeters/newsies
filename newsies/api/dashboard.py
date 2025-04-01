"""
newsies/api.dashboard
"""

import pickle
from typing import List

from dash import Dash, html

import dash_cytoscape as cyto

# pylint: disable=broad-exception-caught

DASHBOARD_APP = Dash(__name__, requests_pathname_prefix="/dashboard/")


def get_knn_graph_data() -> List:
    """get_knn_graph_data"""
    cluster_colors = [
        "aqua",
        "black",
        "blue",
        "blueviolet",
        "cadetblue",
        "crimson",
        "darkcyan",
        "fuchsia",
        "gold",
        "goldenrod",
        "gray",
        "green",
        "lime",
        "maroon",
        "navy",
        "olive",
        "purple",
        "red",
        "silver",
        "teal",
    ]

    elements = []
    edges = []

    try:
        with open("./daily_news/apnews.com/knn.pkl", "rb") as pkl:
            grph = pickle.load(pkl)

        for node in grph.nodes():
            cluster_id = grph.nodes[node].get("cluster", 0)
            color = cluster_colors[cluster_id % len(cluster_colors)]
            # print(f"giving cluster {cluster_id} color {color}")

            pos = grph.nodes[node].get("position", [0, 0])  # Get assigned position

            elements.append(
                {
                    "data": {"id": node, "label": f"Article {node}"},
                    "position": {
                        "x": pos[0],
                        "y": pos[1],
                    },  # Scale for Cytoscape
                    "style": {"background-color": color},
                }
            )

        for u, v in grph.edges:
            g1 = cluster_colors[grph.nodes[u]["cluster"] % len(cluster_colors)]
            g2 = cluster_colors[grph.nodes[v]["cluster"] % len(cluster_colors)]
            edges.append(
                {
                    "data": {"source": str(u), "target": str(v)},
                    "style": {
                        "line-fill": "linear-gradient",
                        "line-style": "dotted" if g1 != g2 else "solid",
                        "width": 3 if g1 == g2 else 2,
                        "line-gradient-stop-colors": f"{g2} {g1}",
                    },
                }
            )

    except Exception:
        pass

    return elements + edges


DASHBOARD_APP.layout = html.Div(
    [
        html.H1("Article Semantic Clustering"),
        cyto.Cytoscape(
            id="article-graph",
            layout={"name": "preset"},
            style={"width": "100%", "height": "100vh"},
            elements=get_knn_graph_data(),
            stylesheet=[
                {"selector": "node", "style": {"label": ""}},
            ],
            responsive=True,
        ),
    ]
)
