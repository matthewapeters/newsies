"""
newsies/api.dashboard
"""

import pickle
from typing import List

from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_cytoscape as cyto

# pylint: disable=broad-exception-caught

# Load extended cytoscape layouts
#  external layouts are:
#  [cose-bilkent](https://github.com/cytoscape/cytoscape.js-cose-bilkent),
#  [fcose](https://github.com/iVis-at-Bilkent/cytoscape.js-fcose),
#  [cola](https://github.com/cytoscape/cytoscape.js-cola),
#  [euler](https://github.com/cytoscape/cytoscape.js-dagre),
#  [spread](https://github.com/cytoscape/cytoscape.js-spread),
#  [dagre](https://github.com/cytoscape/cytoscape.js-dagre),
#  [klay](https://github.com/cytoscape/cytoscape.js-klay),.
cyto.load_extra_layouts()  # Dash Layout

GRAPH_LAYOUT_NAME = "fcose"
LAYOUT_DETAILS = {
    "fcose": {
        "nodeSeparation": 60,  # 80,  # Increase the space between nodes
        "componentSpacing": 300,  # 150,  # Increase the space between clusters
        "gravity": 0.0125,  # 0.25,  # Adjust to pull nodes closer or further from center
        "idealEdgeLength": 60,  # 100,  # Set ideal edge length
        "edgeElasticity": 0.5,  # Adjust elasticity of edges
    },
    "cola": {
        "nodeSpacing": 100,  # Increase spacing between nodes
        "edgeLength": 100,  # Adjust the length of the edges
        "handleScale": False,  # Disable scaling of nodes in the layout
    },
}
DASHBOARD_APP = Dash(__name__, requests_pathname_prefix="/dashboard/")

#
# def get_knn_graph_data() -> List:
#     """get_knn_graph_data"""
#     cluster_colors = [
#         "red",
#         "blue",
#         "green",
#         "purple",
#         "orange",
#         "pink",
#         "cyan",
#         "yellow",
#     ]
#     elements = []
#     edges = []
#     try:
#         with open("./daily_news/apnews.com/knn.pkl", "rb") as pkl:
#             grph = pickle.load(pkl)
#
#         for node in grph.nodes():
#             cluster_id = grph.nodes[node].get("cluster", 0)  # Default to cluster 0
#             color = cluster_colors[
#                 cluster_id % len(cluster_colors)
#             ]  # Assign color based on cluster
#
#             elements.append(
#                 {
#                     "data": {"id": node, "label": f"Article {node}"},
#                     "style": {"background-color": color},
#                 }
#             )
#         for u, v in grph.edges:
#             edges.append({"data": {"source": str(u), "target": str(v)}})
#
#     except Exception:
#         pass
#     return elements + edges


def get_knn_graph_data() -> List:
    """get_knn_graph_data"""
    cluster_colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "pink",
        "cyan",
        "yellow",
    ]
    elements = []
    edges = []

    try:
        with open("./daily_news/apnews.com/knn.pkl", "rb") as pkl:
            grph = pickle.load(pkl)

        for node in grph.nodes():
            cluster_id = grph.nodes[node].get("cluster", 0)
            color = cluster_colors[cluster_id % len(cluster_colors)]

            pos = grph.nodes[node].get("position", [0, 0])  # Get assigned position

            elements.append(
                {
                    "data": {"id": node, "label": f"Article {node}"},
                    "position": {
                        "x": pos[0] * 100,
                        "y": pos[1] * 100,
                    },  # Scale for Cytoscape
                    "style": {"background-color": color},
                }
            )

        for u, v in grph.edges:
            edges.append({"data": {"source": str(u), "target": str(v)}})

    except Exception:
        pass

    return elements + edges


DASHBOARD_APP.layout = html.Div(
    [
        html.H1("Article Clustering"),
        cyto.Cytoscape(
            id="article-graph",
            layout={"name": GRAPH_LAYOUT_NAME, **LAYOUT_DETAILS[GRAPH_LAYOUT_NAME]},
            style={"width": "100%", "height": "60vh"},
            elements=get_knn_graph_data(),
            stylesheet=[
                {"selector": "node", "style": {"label": ""}},
                {"selector": "edge", "style": {"width": 1, "line-color": "#888"}},
            ],
            responsive=True,
        ),
        # Sliders to control layout settings
        html.Div(
            [
                html.Label("Ideal Edge Length"),
                dcc.Slider(
                    id="edge-length-slider",
                    min=50,
                    max=500,
                    step=10,
                    value=100,
                    marks={i: str(i) for i in range(50, 501, 50)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Label("Node Separation"),
                dcc.Slider(
                    id="node-separation-slider",
                    min=50,
                    max=200,
                    step=10,
                    value=100,
                    marks={i: str(i) for i in range(50, 201, 50)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Label("Node Repulsion"),
                dcc.Slider(
                    id="node-repulsion-slider",
                    min=50,
                    max=100000,
                    step=10,
                    value=100000,
                    marks={i: str(i) for i in range(50, 1000000, 50)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Label("Component Spacing"),
                dcc.Slider(
                    id="component-spacing-slider",
                    min=50,
                    max=1000,
                    step=10,
                    value=500,
                    marks={i: str(i) for i in range(50, 1001, 50)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Label("Gravity"),
                dcc.Slider(
                    id="gravity-slider",
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.1,
                    marks={i / 10: str(i / 10) for i in range(0, 11)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Label("Edge Elasticity"),
                dcc.Slider(
                    id="edge-elasticity-slider",
                    min=0,
                    max=1,
                    step=0.1,
                    value=0.1,
                    marks={i / 10: str(i / 10) for i in range(0, 10, 2)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Label("Nesting Factor"),
                dcc.Slider(
                    id="nesting-factor-slider",
                    min=0,
                    max=1,
                    step=0.1,
                    value=0.1,
                    marks={i / 10: str(i / 10) for i in range(0, 10, 2)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ],
            style={"padding": "2px"},
        ),
    ]
)


# Update Cytoscape layout when slider values change
@DASHBOARD_APP.callback(
    Output("article-graph", "layout"),
    [
        Input("node-separation-slider", "value"),
        Input("component-spacing-slider", "value"),
        Input("gravity-slider", "value"),
        Input("edge-length-slider", "value"),
        Input("edge-elasticity-slider", "value"),
        Input("node-repulsion-slider", "value"),
        Input("nesting-factor-slider", "value"),
    ],
)
def update_layout(
    node_separation,
    component_spacing,
    gravity,
    ideal_edge_length,
    edge_elasticity,
    node_repulsion,
    nesting_factor,
):
    """update_layout"""

    print(f"Node Separation: {node_separation}")
    print(f"Node Repulsion: {node_repulsion}")
    print(f"Component Spacing: {component_spacing}")
    print(f"Gravity: {gravity}")
    print(f"Ideal Edge Length: {ideal_edge_length}")
    print(f"Edge Elasticity Length: {edge_elasticity}")
    print(f"Nesting Factor: {nesting_factor}")

    return {
        "name": "fcose",
        "nodeSeparation": node_separation,
        "nodeRepulsion": node_repulsion,
        "componentSpacing": component_spacing,
        "gravity": gravity,
        "idealEdgeLength": ideal_edge_length,
        "edgeElasticity": edge_elasticity,
        "nestingFactor": nesting_factor,
        "damping": 0.5,
        "randomize": True,
        "packing": True,
    }
