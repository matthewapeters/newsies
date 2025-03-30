"""
newsies.articles
    tools for accessing and working with the cached articles
"""

import os
import pickle
from typing import Any, Dict, List, Tuple, Union
import math

from collections import Counter

import networkx as nx
import community  # Louvain clustering
from chromadb import Collection
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

from newsies.collections import NEWS
from newsies.ap_news.article import Article
from newsies.chroma_client import ChromaDBClient

ARCHIVE = f"./daily_news/{NEWS}"


class Archive:
    """
    Archive
        current news archive
    """

    def __init__(self):
        self.collection: Dict[str, Union[Article, Any]] = {}
        self.graph: nx.Graph = None
        self.archive_path = ARCHIVE
        self.cluster_index = {}
        self.cluster_profile: pd.DataFrame = None
        self.ner_counts: Counter = Counter()

    def refresh(self):
        """
        refresh
            get the latest list of articles in the ARCHIVE
        """
        for filename in os.listdir(self.archive_path):
            item_id = filename[:-4]
            # remove non-article files from the collection
            if item_id in ["knn", "latest_news"]:
                continue
            if item_id not in self.collection:
                self.collection[item_id] = ""

    def get_article(self, item_id: str):
        """
        get_article
        """
        file_name = item_id
        if ".pkl" in item_id:
            item_id = file_name[:-4]
        else:
            file_name = f"{item_id}.pkl"

        if not isinstance(self.collection[item_id], Article):
            with open(f"{self.archive_path}/{file_name}", "rb") as pikl:
                self.collection[item_id] = pickle.load(pikl)
        return self.collection[item_id]

    def build_knn(self):
        """build_knn"""
        cdb = ChromaDBClient()
        cdb.collection_name = NEWS
        keys = self.collection.keys()
        network = [{k: get_nearest_neighbors(cdb.collection, k, 5)} for k in keys]

        self.graph, partitions = build_similarity_graph(network=network)

        # Assign clusters
        for node, cluster in partitions.items():
            self.graph.nodes[node]["cluster"] = cluster

        # Assign initial positions
        self.graph = initialize_positions(self.graph, partitions)

        # Save graph with positions
        with open(f"{self.archive_path}/knn.pkl", "wb") as pkl:
            pickle.dump(self.graph, pkl)

    def load_graph(self):
        """load_graph"""
        with open(f"{self.archive_path}/knn.pkl", "rb") as pkl:
            self.graph = pickle.load(pkl)

    def load_cluster_profiles(self):
        """load_metadatas"""
        client = ChromaDBClient()
        client.collection_name = NEWS
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


def get_nearest_neighbors(
    collection: Collection, article_id: str, k: int = 5
) -> Dict[str, Union[List[Tuple[str, float]], List[float]]]:
    """
    Query ChromaDB to find k nearest articles by embedding similarity.
    """
    article_data = collection.get(ids=[article_id], include=["embeddings"])

    # Ensure embeddings exist and are not empty
    if (
        not article_data
        or "embeddings" not in article_data
        or not article_data["embeddings"].any()
    ):
        print(f"Warning: No embedding found for article_id {article_id}")
        return []

    embeddings = article_data["embeddings"][0]  # Extract stored embedding
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
        details = node[article_id]
        grph.add_node(article_id)  # Add article as a node
        neighbors = details["neighbors"]
        node_embeddings = details["embeddings"]
        grph.nodes[article_id]["embeddings"] = node_embeddings

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


def initialize_positions(grph: nx.Graph, partition: Dict[str, int]):
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
    cluster_order.sort(key=lambda x: len(clusters[x]))
    for n in grph.nodes:
        c = grph.nodes[n]["cluster"]
        c_new = cluster_order.index(c)
        grph.nodes[n]["cluster"] = c_new

    print("cluster_order: ", [f"{x}: {len(clusters[x])}" for x in cluster_order])
    max_cluster_size = len(clusters[cluster_order[-1]])
    # rebuild clusters, indexed on size
    new_clusters = {i: clusters[c] for i, c in enumerate(cluster_order)}
    clusters = new_clusters

    # compute positions based on radial cluster centroids
    cluster_angle = math.sqrt(len(clusters)) * math.pi / len(clusters)

    nx.set_node_attributes(
        grph, partition, "cluster"
    )  # Add clusters as node attributes

    for cluster_id, nodes in clusters.items():
        cluster_size = len(nodes)
        node_angle = 2 * math.pi / cluster_size
        cluster_center = (
            math.cos(cluster_angle * cluster_id) * (max_cluster_size * cluster_id),  # x
            math.sin(cluster_angle * cluster_id) * (max_cluster_size * cluster_id),  # y
        )
        for i, node in enumerate(nodes):
            node_x = cluster_center[0] + (
                math.cos(node_angle * i) * (cluster_size * max_cluster_size / 16)
            )
            node_y = cluster_center[1] + (
                math.sin(node_angle * i) * (cluster_size * max_cluster_size / 16)
            )
            grph.nodes[node]["position"] = (node_x, node_y)
            # only keep edges between cluster-mates
            # edges_to_prune = []
            # for e in grph.edges(node):
            #     if e[0] == node:
            #         if e[1] not in clusters[cluster_id]:
            #             edges_to_prune.append(e)
            #     else:
            #         if e[0] not in clusters[cluster_id]:
            #             edges_to_prune.append(e)
            # grph.remove_edges_from(edges_to_prune)

    return grph
