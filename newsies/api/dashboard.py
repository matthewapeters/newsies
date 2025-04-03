"""
newsies/api.dashboard
"""

import pickle
from typing import List

import networkx as nx

from dash import Dash, html, Input, Output

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

    els = []
    ejs = []
    ej_clazs = set()  # Track unique classes to generate styles
    nl_clazs = set()
    grph: nx.Graph = None
    try:
        with open("./daily_news/apnews.com/knn.pkl", "rb") as pkl:
            grph = pickle.load(pkl)

        for node in grph.nodes():
            cluster_id = grph.nodes[node].get("cluster", 0)
            color = cluster_colors[cluster_id % len(cluster_colors)]
            # print(f"giving cluster {cluster_id} color {color}")

            pos = grph.nodes[node].get("position", [0, 0])  # Get assigned position

            els.append(
                {
                    "data": {"id": node, "label": f"Article {node}"},
                    "position": {
                        "x": pos[0],
                        "y": pos[1],
                    },  # Scale for Cytoscape
                    "classes": f"node-{color}",
                }
            )
            nl_clazs.add(f"node-{color}")

        for u, v in grph.edges:
            g1 = cluster_colors[grph.nodes[u]["cluster"] % len(cluster_colors)]
            g2 = cluster_colors[grph.nodes[v]["cluster"] % len(cluster_colors)]
            ej_claz = f"edge-{g1}-{g2}" if g1 != g2 else f"edge-{g1}"

            ejs.append(
                {
                    "data": {
                        "source": str(u),
                        "target": str(v),
                    },
                    "classes": ej_claz,
                }
            )
            ej_clazs.add(ej_claz)

    except Exception as e:
        print(f"EXCEPTION: {e} u:{u} v: {v} g1: {g1} g2: {g2}")

    return els + ejs, ej_clazs, nl_clazs


elements, edge_classes, node_classes = get_knn_graph_data()


def generate_styles(edge_classes_set, node_classes_set):
    """Generate stylesheet dynamically for edges"""
    return (
        [
            {
                "selector": f".{clz}",
                "style": {"label": "", "background-color": clz.split("-")[1]},
            }
            for clz in node_classes_set
        ]
        + [
            {
                "selector": f".{cls}",
                "style": {
                    "line-style": "solid",
                    "width": 3,
                    "line-color": cls.split("-")[1],
                },
            }
            for cls in edge_classes_set
            if len(cls.split("-")) < 3
        ]
        + [
            {
                "selector": f".{cls}",
                "style": {
                    "line-style": "dashed",
                    "width": 2,
                    "line-fill": "linear-gradient",
                    "line-gradient-stop-colors": f"{cls.split("-")[2]} {cls.split("-")[1]}",
                },
            }
            for cls in edge_classes_set
            if len(cls.split("-")) == 3
        ]
    )


DASHBOARD_APP.layout = html.Div(
    [
        html.H1("Article Semantic Clustering"),
        html.Button("Refresh", id="refresh-btn", n_clicks=0),
        cyto.Cytoscape(
            id="article-graph",
            layout={"name": "preset"},
            style={"width": "100%", "height": "100vh"},
            elements=elements,  # Initial elements
            stylesheet=generate_styles(
                edge_classes, node_classes
            ),  # Initial stylesheet
            responsive=True,
        ),
    ]
)


@DASHBOARD_APP.callback(
    [Output("article-graph", "elements"), Output("article-graph", "stylesheet")],
    Input("refresh-btn", "n_clicks"),
)
def update_graph(_):
    """Update graph elements and stylesheet dynamically"""
    e, ec, nc = get_knn_graph_data()
    return e, generate_styles(ec, nc)
