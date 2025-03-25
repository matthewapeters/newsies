"""
newsies.articles
    tools for accessing and working with the cached articles
"""

import os
import pickle
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import networkx as nx

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

    def refresh(self):
        """
        refresh
            get the latest list of articles in the ARCHIVE
        """
        for filename in os.listdir(ARCHIVE):
            item_id = filename[:-4]
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
            with open(f"{ARCHIVE}/{file_name}", "rb") as pikl:
                self.collection[item_id] = pickle.load(pikl)
        return self.collection[item_id]

    def build_knn(self):
        """
        build_knn
        """
        cdb = ChromaDBClient()
        cdb.collection_name = NEWS
        keys = self.collection.keys()
        network = [{k: get_nearest_neighbors(cdb.collection, k, 5)} for k in keys]


def get_nearest_neighbors(collection, article_id, k=5) -> List[Tuple[str, float]]:
    """
    Query ChromaDB to find k nearest articles by embedding similarity.
    """
    article_data = collection.get(ids=[article_id], include=["embeddings"])

    if not article_data["embeddings"]:
        return []

    embedding = article_data["embeddings"][0]  # Extract stored embedding
    results = collection.query(
        query_embeddings=[embedding], n_results=k + 1
    )  # k+1 to exclude self-match

    neighbors = list(
        zip(results["ids"][0], results["distances"][0])
    )  # Extract neighbor IDs and distances
    neighbors = [
        (nid, dist) for nid, dist in neighbors if nid != article_id
    ]  # Remove self-match

    return neighbors  # List of (neighbor_id, similarity_score)
