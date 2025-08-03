"""
newsies/api.dashboard
"""

import pickle
from typing import Callable, List

import networkx as nx

from dash import Dash, dcc, html, Input, Output, callback_context

import dash_cytoscape as cyto

from ..ap_news.archive import Archive

# pylint: disable=broad-exception-caught

DASHBOARD_APP = Dash(__name__, requests_pathname_prefix="/dashboard/")
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


def get_knn_graph_data(get_data: Callable, *args) -> List:
    """get_knn_graph_data"""

    els = []
    ejs = []
    ej_clazs = set()  # Track unique classes to generate styles
    nl_clazs = set()
    grph: nx.Graph = None
    u: str = None
    v: str = None
    g1: str = None
    g2: str = None
    try:
        grph = get_data(*args)

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


def get_knn_graph() -> nx.Graph:
    """get_knn_graph"""
    grph: nx.Graph = None
    try:
        with open("./daily_news/apnews.com/knn.pkl", "rb") as pkl:
            grph = pickle.load(pkl)
    except Exception as e:
        print(f"EXCEPTION: {e}")
    return grph


def get_most_recent_graph(recency_offset: int = 0) -> nx.Graph:
    """get_most_recent_graph"""
    grph = get_knn_graph()
    if grph is None:
        return nx.Graph()  # Ensure it always returns a valid Graph
    return grph.subgraph(Archive.most_recent_articles(recency_offset))


offset: int = 0
elements, edge_classes, node_classes = get_knn_graph_data(get_data=get_knn_graph)
most_recent_elements, mr_edge_classes, mr_node_classes = get_knn_graph_data(
    get_most_recent_graph, offset
)


def generate_styles(edge_classes_set, node_classes_set):
    """Generate stylesheet dynamically for edges"""
    return (
        [
            {
                "selector": f".{clz}",
                "style": {"label": "", "background-color": clz.split("-")[1]},
            }
            for clz in node_classes_set
            if "-" in clz
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
            if "-" in cls and len(cls.split("-")) < 3
        ]
        + [
            {
                "selector": f".{cls}",
                "style": {
                    "line-style": "dashed",
                    "width": 2,
                    "line-fill": "gradient",
                },
            }
            for cls in edge_classes_set
            if "-" in cls and len(cls.split("-")) == 3
        ]
    )


DASHBOARD_APP.layout = html.Div(
    [
        html.H1("Article Semantic Clustering"),
        html.Div(
            [
                html.Table(
                    [
                        html.Tr(
                            [
                                html.Td(
                                    html.Button(
                                        "Refresh", id="refresh-btn", n_clicks=0
                                    ),
                                    style={"width": "50%"},
                                ),
                                html.Td(
                                    [
                                        html.H5("Recently Published Date Offset"),
nodes.append(
    {
        "data": {
            "id": doc_id,
            "label": title[:50] + "..." if len(title) > 50 else title,
            "title": title,
            "url": url,
        }
    }
)
                                        dcc.Slider(
                                            id="offset-slider",
                                            min=0,
                                            max=90,
                                            step=1,
                                            value=offset,
                                            marks={i: str(i) for i in range(0, 91, 5)},
                                        ),
                                    ],
                                    style={"width": "50%"},
                                ),
                            ],
                            style={"width": "100%"},
                        ),
                    ],
                    style={
                        "width": "98%",
                        "border": "1px solid black",
                        "padding": "10px",
                    },
                ),
                html.Div(
                    [
                        html.H2("Article Graph"),
                        cyto.Cytoscape(
                            id="article-graph",
                            layout={"name": "preset"},
                            style={"width": "100%", "height": "80vh"},
                            elements=elements,
                            stylesheet=generate_styles(edge_classes, node_classes),
                            responsive=True,
                        ),
                    ],
                    style={
                        "width": "49%",
                        "display": "inline-block",
                        "vertical-align": "top",
                        "border": "1px solid black",
                    },
                ),
                html.Div(
                    [
                        html.H2("Most Recent Articles"),
                        cyto.Cytoscape(
                            id="most-recent",
                            layout={"name": "preset"},
                            style={"width": "100%", "height": "80vh"},
                            elements=most_recent_elements,
                            stylesheet=generate_styles(
                                mr_edge_classes, mr_node_classes
                            ),
                            responsive=True,
                        ),
                    ],
                    style={
                        "width": "49%",
                        "display": "inline-block",
                        "vertical-align": "top",
                        "border": "1px solid black",
                    },
                ),
            ]
        ),
    ]
)


# pylint: disable=global-statement


@DASHBOARD_APP.callback(
    Output("article-graph", "elements"),
    Output("article-graph", "stylesheet"),
    Output("most-recent", "elements"),
    Output("most-recent", "stylesheet"),
    Output("offset-slider", "value"),
    Input("refresh-btn", "n_clicks"),
    Input("offset-slider", "value"),
    prevent_initial_call=True,
)
def update_graph(_, slider_value):
    """
    Update graph elements and stylesheet dynamically
    Handle both refresh and slider offset
    """
    # Determine if refresh was clicked
    triggered = [t["prop_id"] for t in callback_context.triggered]

    global offset
    if "refresh-btn.n_clicks" in triggered:
        offset = 0
    else:
        offset = slider_value or 0

    # Re-fetch both graphs
    e, ec, nc = get_knn_graph_data(get_data=get_knn_graph)
    mre, mrec, mrnc = get_knn_graph_data(get_most_recent_graph, offset)

    return (
        e,
        generate_styles(ec, nc),
        mre,
        generate_styles(mrec, mrnc),
        offset,
    )
