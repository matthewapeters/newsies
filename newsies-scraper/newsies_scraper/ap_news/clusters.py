"""
newsies.ap_news.clusters

    For clustering articles on semantic similarity.
"""

from collections import Counter
import math
import pickle
from typing import Any, Dict, List, Tuple, Union

from chromadb import Collection
import community  # Louvain clustering
import networkx as nx
import numpy as np
import pandas as pd

from ..ap_news.article import Article
from newsies_clients.chroma_client import ChromaDBClient
from newsies_clients.collections import NEWS

from .semaphores import protected, get_protected

ARCHIVE_PATH = f"./daily_news/{NEWS}"
CLUSTER: "Cluster" = None

# pylint: disable=global-statement, broad-exception-caught


@get_protected
def get_cluster() -> "Cluster":
    """
    get_archive
        returns the Cluster singleton
    """
    global CLUSTER
    if CLUSTER is None:
        try:
            CLUSTER = Cluster.load()
        except Exception as e:
            print(f"cannot load archive ({e}) -- initiating new archive")
            CLUSTER = Cluster()
            CLUSTER.dump()
    return CLUSTER


class Cluster:
    """Cluster"""

    def __init__(self):
        self.graph: nx.Graph = None
        self.most_recent_graph: nx.Graph = None
        # Louvaine clusters
        self.batches: List[List[str]] = []
        self.clusters: Dict[int, List[str]]
        self.cluster_index = {}
        self.archive_path: str = ""
        self.collection: Dict[str, Union[Article, Any]] = {}
        self.cluster_profile: pd.DataFrame = None
        self.ner_counts: Counter = Counter()

    @protected
    def to_dict(self) -> str:
        """to_dict"""
        struct = {
            "archive_path": self.archive_path,
            "ner_counts": self.ner_counts.most_common(),
            "batches": self.batches,
            "clusters": self.clusters,
        }
        return struct

    @staticmethod
    def load() -> "Cluster":
        """
        load
            loads the last pickled backup of the archive
            will throw a FileNotFound if the archive.pkl
            is not present
        """
        with open(f"{ARCHIVE_PATH}/knn_cluster.pkl", "rb") as pkl:
            return pickle.load(pkl)

    @protected
    def dump(self):
        """
        dump
            backs up the archive as a pickled object
        """
        with open(f"{self.archive_path}/knn_cluster.pkl", "wb") as pkl:
            pickle.dump(self, pkl)

    def _initialize_positions(self, partition: Dict[str, int]):
        """
        position cluster centers around center,
        and position nodes around cluster centers
        similar to CiSE
        """
        clusters = {
            p: [nn for nn, pp in partition.items() if pp == p]
            for p in set(partition.values())
        }

        # re-order clusters by size
        cluster_order = list(clusters.keys())
        cluster_order.sort(key=lambda x: (len(clusters[x]) * 10000) + x)
        for n in self.graph.nodes:
            c = self.graph.nodes[n]["cluster"]
            c_new = cluster_order.index(c)
            self.graph.nodes[n]["cluster"] = c_new

        # print("cluster_order: ", [f"{x}: {len(clusters[x])}" for x in cluster_order])
        max_cluster_size = len(clusters[cluster_order[-1]])

        # rebuild clusters, indexed on size
        new_clusters = {i: clusters[c] for i, c in enumerate(cluster_order)}
        self.clusters = new_clusters

        # Add clusters as node attributes
        nx.set_node_attributes(
            self.graph, {n: k for k, v in self.clusters.items() for n in v}, "cluster"
        )

        # create groups (based on clusters where members 2 or more)
        groups: List[List[str]] = [v for v in self.clusters.values() if len(v) > 1]
        # create a group (singles) of clusters with only one member
        singles = [
            single_node
            for indy_group in self.clusters.values()
            if len(indy_group) == 1
            for single_node in indy_group
        ]
        # add the singles group to groups
        groups.append(singles)

        # sort groups by member count
        groups.sort(key=len, reverse=False)

        # the groups will be used for incremental model training batches
        self.batches = groups

        # compute positions based on radial group centroids
        group_angle = math.sqrt(len(groups)) * math.pi / len(groups)

        for group_id, nodes in enumerate(groups):
            group_size = len(nodes)
            node_angle = 2 * math.pi / group_size
            cluster_center = (
                math.cos(group_angle * group_id)
                * (max_cluster_size * max(group_id, math.sqrt(len(nodes)))),  # x
                math.sin(group_angle * group_id)
                * (max_cluster_size * max(group_id, math.sqrt(len(nodes)))),  # y
            )
            for i, node in enumerate(nodes):
                node_x = cluster_center[0] + (
                    math.cos(node_angle * i) * (group_size * max_cluster_size / 16)
                )
                node_y = cluster_center[1] + (
                    math.sin(node_angle * i) * (group_size * max_cluster_size / 16)
                )
                self.graph.nodes[node]["position"] = (node_x, node_y)

        for n1, n2 in self.graph.edges:
            weight = self.graph.edges[n1, n2]["weight"]
            c1 = self.graph.nodes[n1]["cluster"]
            c2 = self.graph.nodes[n2]["cluster"]
            edge_class = {"weight": weight, "clusters": [c1, c2]}
            self.graph.edges[n1, n2]["edge_class"] = edge_class

    @protected
    def build_knn(self):
        """build_knn"""
        cdb = ChromaDBClient()
        cdb.collection_name = NEWS
        keys = self.collection.keys()
        network = [
            {k: get_nearest_neighbors(cdb.collection, k, 5)}
            for k in keys
            if len(k) == 32
        ]

        self.graph, partitions = build_similarity_graph(network=network)

        # Assign clusters
        for node, cluster in partitions.items():
            self.graph.nodes[node]["cluster"] = cluster

        # Assign initial positions
        self._initialize_positions(partitions)

        # Save graph with positions
        with open(f"{self.archive_path}/knn.pkl", "wb") as pkl:
            pickle.dump(self.graph, pkl)

    @protected
    def load_graph(self):
        """load_graph"""
        with open(f"{self.archive_path}/knn.pkl", "rb") as pkl:
            self.graph = pickle.load(pkl)

    @protected
    def load_cluster_profiles(self):
        """load_metadatas"""
        if self.graph is None:
            self.load_graph()
        for node in self.graph.nodes():
            a = Article.load(self.archive_path, node)
            self.graph.nodes[node]["embeddings"] = a.embeddings
            self.ner_counts.update(a.named_entities)
            self.ner_counts.update(a.section_headlines.keys())
            del a

        data = [
            {
                "item_id": node,
                "cluster": attrs["cluster"],
                **{f"dim{i:03}": e for i, e in enumerate(attrs["embeddings"])},
            }
            for node, attrs in self.graph.nodes(data=True)
        ]
        self.cluster_profile = pd.DataFrame(data)


def position_clusters(
    clusters,
    graph,
    base_radius=5,
    expansion_factor=3,
    angle_increment=np.radians(137.5),
):
    """
    This came from chatGPT - needs to be validated -
    my solution works better at the moment!

    Positions clusters using a logarithmic spiral and arranges nodes in each cluster
    in a circle around the centroid.

    Parameters:
        clusters (List[List[node]]): List of clusters, where each cluster is a list of
        node identifiers.
        graph (nx.Graph): A NetworkX graph containing all nodes.
        base_radius (float): Minimum radial distance for the first cluster.
        expansion_factor (float): Controls outward spiral growth.
        angle_increment (float): Base angle increment (default is the golden angle).
    """

    # Sort clusters by size (smallest first)
    clusters = sorted(clusters, key=len)

    # Compute cluster radii based on node count
    cluster_radii = [np.sqrt(len(cluster)) for cluster in clusters]

    # Positioning variables
    theta = 0
    node_positions = {}

    for i, cluster in enumerate(clusters):
        # Compute cluster centroid position using logarithmic spiral
        radius = base_radius + expansion_factor * np.log(1 + i)
        cx, cy = radius * np.cos(theta), radius * np.sin(theta)

        # Distribute nodes in a circle around the centroid
        num_nodes = len(cluster)
        node_angle_step = 2 * np.pi / num_nodes

        for j, node in enumerate(cluster):
            angle = j * node_angle_step
            node_x = cx + cluster_radii[i] * np.cos(angle)
            node_y = cy + cluster_radii[i] * np.sin(angle)
            node_positions[node] = (node_x, node_y)

        # Move to the next angle in the spiral
        theta += angle_increment

    # Set positions in the graph
    nx.set_node_attributes(graph, node_positions, "pos")


def get_nearest_neighbors(
    collection: Collection, article_id: str, k: int = 5
) -> Dict[str, Union[List[Tuple[str, float]], List[float]]]:
    """
    Query ChromaDB to find k nearest articles by embedding similarity.
    """
    article_data = collection.get(ids=[article_id], include=["embeddings", "metadatas"])

    # Ensure embeddings exist and are not empty
    if (
        not article_data
        or "embeddings" not in article_data
        or not article_data["embeddings"].any()
    ):
        print(f"Warning: No embedding found for article_id {article_id}")
        if len(article_id) == 32:
            with open(
                f"{ARCHIVE_PATH}/missing_embeddings.txt", "a", encoding="utf8"
            ) as missing_file:
                missing_file.write(f"{article_id}\n")
            return []

    embeddings = article_data["embeddings"][0]  # Extract stored embedding
    metadatas = article_data["metadatas"][0]
    results = collection.query(
        query_embeddings=[embeddings], n_results=k + 1
    )  # k+1 to exclude self-match

    neighbors = list(
        zip(results["ids"][0], results["distances"][0])
    )  # Extract neighbor IDs and distances
    neighbors = [
        (nid, dist) for nid, dist in neighbors if nid != article_id
    ]  # Remove self-match

    return {
        "neighbors": neighbors,
        "embeddings": embeddings,
        "metadatas": metadatas,
    }  # Dict neighbors: List(neighbor_id, similarity_score), embeddings: embeddings


def build_similarity_graph(
    network: List[Dict[str, List[Tuple[str, float]]]], similarity_threshold=0.8
) -> Tuple[nx.Graph, Dict[str, int]]:
    """
    Constructs a NetworkX graph where articles are nodes and edges are similarity-based connections.
    """
    grph = nx.Graph()

    embeddings = {}  # Store embeddings for repulsion logic

    for node in network:
        article_id = list(node.keys())[0]
        # Check if article_id is valid
        if len(article_id) != 32:
            continue
        details = node[article_id]
        grph.add_node(article_id)  # Add article as a node
        neighbors = details["neighbors"]
        node_embeddings = details["embeddings"]
        grph.nodes[article_id]["embeddings"] = node_embeddings
        publish_date = details["metadatas"]["publish_date"]
        grph.nodes[article_id]["publish_date"] = publish_date

        # Retrieve nearest neighbors
        for neighbor_id, similarity in neighbors:
            if (
                similarity >= similarity_threshold
            ):  # Filter edges by similarity threshold
                grph.add_edge(article_id, neighbor_id, weight=similarity)

    # Assign clusters
    partition = community.best_partition(grph)  # {node_id: cluster_id}

    # Assign node embeddings (for positioning)
    for node in grph.nodes():
        embeddings[node] = np.random.rand(2)  # Temporary random 2D positions

    return grph, partition
