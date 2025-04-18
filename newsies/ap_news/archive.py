"""
newsies.articles
    tools for accessing and working with the cached articles
"""

from functools import wraps

from typing import Any, Callable, Dict, List, Set, Tuple, Union
import math
import os
import pickle
import threading
import json
from multiprocessing import Lock as MpLock

from collections import Counter

import networkx as nx
import community  # Louvain clustering
from chromadb import Collection
import pandas as pd
import numpy as np

from newsies.collections import NEWS
from newsies.redis_client import REDIS
from newsies.ap_news.article import Article
from newsies.chroma_client import ChromaDBClient

# pylint: disable=global-statement, too-many-locals, consider-using-with, broad-exception-caught

ARCHIVE_PATH = f"./daily_news/{NEWS}"
ARCHIVE: "Archive" = None
_MTX: threading.Lock = threading.Lock()
_GET_LOCK: threading.Lock = threading.Lock()
_MP_LOCK = MpLock()


def protected_factory(mtx: threading.Lock = _MTX) -> Callable:
    """
    protected wrapper
        Wraps functions with MTX lock
        param: mtx: Lock -- which mutex to use.  Defaults to _MTX
    """

    def wrapper_factory(c: Callable):
        """wrapper_factory"""

        @wraps(c)
        def protected_callable(*args, **kwargs):
            """protected_callable"""
            with mtx:
                return c(*args, **kwargs)

        return protected_callable

    return wrapper_factory


protected = protected_factory()
get_protected = protected_factory(_GET_LOCK)
multi_processing_protected = protected_factory(_MP_LOCK)


@get_protected
def get_archive() -> "Archive":
    """
    get_archive
        returns the Archive singleton
    """
    global ARCHIVE
    if ARCHIVE is None:
        try:
            ARCHIVE = Archive.load()
        except Exception as e:
            print(f"cannot load archive ({e}) -- initiating new archive")
            ARCHIVE = Archive()
            ARCHIVE.dump()
    return ARCHIVE


class Archive:
    """
    Archive
        current news archive
    """

    def __init__(self):
        self.collection: Dict[str, Union[Article, Any]] = {}
        self.graph: nx.Graph = None
        self.most_recent_graph: nx.Graph = None
        self.archive_path = ARCHIVE_PATH
        self.cluster_index = {}
        self.cluster_profile: pd.DataFrame = None
        self.ner_counts: Counter = Counter()

        # Louvaine clusters
        self.clusters: Dict[int, List[str]]

        # clusters ordered for training
        self.batches: List[List[str]] = []
        self.trained_batches: Set[Tuple[str]] = set()

        # dicts of model train times and path to LoRA adapter
        self.model_train_dates: Dict[str, str] = {}

    @protected
    def to_dict(self) -> str:
        """to_dict"""
        struct = {
            "archive_path": self.archive_path,
            "ner_counts": self.ner_counts.most_common(),
            "by_publish_dates": self.by_publish_date,
            "model_train_dates": self.model_train_dates,
            "batches": self.batches,
            "clusters": self.clusters,
        }
        return struct

    @protected
    def refresh(self):
        """
        refresh
            get the latest list of articles in the ARCHIVE
        """
        for filename in os.listdir(self.archive_path):
            item_id = filename[:-4]
            # remove non-article files from the collection
            if item_id in ["knn", "latest_news", "archive"]:
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

    @multi_processing_protected
    @staticmethod
    def register_by_publish_date(article: Article):
        """
        register_by_publish_date
            maintain an index of articles by publish date YYYYMMDD
        """
        raw = REDIS.get("by_publish_date") or "{}"
        by_publish_date: Dict = json.loads(raw)
        pub_date = article.publish_date.strftime(r"%Y%m%d")
        if pub_date not in by_publish_date:
            by_publish_date[pub_date] = [article.item_id]
        else:
            if article.item_id not in by_publish_date[pub_date]:
                by_publish_date[pub_date].append(article.item_id)
        REDIS.set("by_publish_date", json.dumps(by_publish_date))

    @staticmethod
    @multi_processing_protected
    def by_publish_date(_=None) -> Dict[str, List[str]]:
        """
        by_publish_date
            threadsafe retrieval of articles grouped by
            publish date (see register_by_publish_date)
        """
        raw = REDIS.get("by_publish_date")
        return json.loads(raw)

    @staticmethod
    def most_recent_articles(offset: int = 0) -> nx.Graph:
        """
        most_recent_articles
            returns the most recent clusters
            from the archive
        """
        # get by_publish_date
        articles_by_publish_date: Dict[str, List[str]] = Archive.by_publish_date()
        dates = list(articles_by_publish_date.keys())
        # sort the dates
        dates.sort(reverse=True)
        # get the most recent date
        most_recent_date = dates[offset]
        # get the articles for that date
        most_recent_articles = articles_by_publish_date.get(most_recent_date, [])
        # filter clusters by the most recent articles
        # and return the most recent clusters
        return most_recent_articles

    @staticmethod
    def publish_dates() -> List[str]:
        """
        publish_dates
            returns the list of dates in the archive
        """
        # get by_publish_date
        articles_by_publish_date: Dict[str, List[str]] = Archive.by_publish_date()
        dates = list(articles_by_publish_date.keys())
        # sort the dates
        dates.sort(reverse=True)
        return dates

    @protected
    def dump(self):
        """
        dump
            backs up the archive as a pickled object
        """
        with open(f"{self.archive_path}/archive.pkl", "wb") as pkl:
            pickle.dump(self, pkl)

    @staticmethod
    def load() -> "Archive":
        """
        load
            loads the last pickled backup of the archive
            will throw a FileNotFound if the archive.pkl
            is not present
        """
        with open(f"{ARCHIVE_PATH}/archive.pkl", "rb") as pkl:
            return pickle.load(pkl)

    @protected
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

    def build_batches(self) -> Dict[int, Tuple[str]]:
        """
        build_batches
            builds the batches of articles for training
            based on the clusters and publish dates
            of the articles in the archive
        """
        if self.trained_batches is None:
            self.trained_batches = set()
        batches: Dict[int, Tuple[str]] = {}
        for offset, pub_date in enumerate(Archive.publish_dates()):
            # get the articles for that date
            most_recent_articles = Archive.most_recent_articles(offset)
            # sort each group alphanumerically so the set will have consistent
            # comparability.
            offset_batches = [
                tuple(set(sorted(n for n in grp if n in most_recent_articles)))
                for grp in self.batches
            ]
            offset_batches = [
                b
                for b in offset_batches
                if b not in self.trained_batches and len(b) > 0
            ]
            batches[pub_date] = offset_batches
        return batches

    def get_articles(self, article_ids: List[str]) -> List[Article]:
        """
        get_articles
            returns the articles for the given article_ids
        """
        articles = []
        for article_id in article_ids:
            if article_id == "None" or article_id == "":
                continue
            if article_id in self.collection and isinstance(
                self.collection[article_id], Article
            ):
                articles.append(self.collection[article_id])
            else:
                a = Article.load(self.archive_path, article_id)
                if a is not None:
                    articles.append(a)
                    self.collection[article_id] = a
        return articles


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
        if article_id == "None" or article_id == "":
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
