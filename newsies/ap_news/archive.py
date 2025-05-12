"""
newsies.articles
    tools for accessing and working with the cached articles
"""

from typing import Any, Dict, List, Tuple, Union
import os
import pickle
import json


import networkx as nx

from newsies.collections import NEWS
from newsies.redis_client import REDIS
from newsies.ap_news.article import Article

from .semaphores import protected, get_protected, multi_processing_protected

# pylint: disable=global-statement, too-many-locals, consider-using-with, broad-exception-caught

ARCHIVE_PATH = f"./daily_news/{NEWS}"
ARCHIVE: "Archive" = None


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
        self.archive_path = ARCHIVE_PATH
        # self.cluster_index = {}
        # self.cluster_profile: pd.DataFrame = None

        # Louvaine clusters
        # self.clusters: Dict[int, List[str]]

        # clusters ordered for training
        self.batches: List[List[str]] = []
        # self.trained_batches: Set[Tuple[str]] = set()

        # dicts of model train times and path to LoRA adapter
        self.model_train_dates: Dict[str, str] = {}

    @protected
    def to_dict(self) -> str:
        """to_dict"""
        struct = {
            "archive_path": self.archive_path,
            "by_publish_dates": self.by_publish_date(),
            "model_train_dates": self.model_train_dates,
            "batches": self.batches,
            #            "clusters": self.clusters,
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
            if item_id in ["knn", "knn_cluster", "latest_news", "archive"]:
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

    def build_batches(self) -> Dict[int, Tuple[str]]:
        """
        build_batches
            builds the batches of articles for training
            based on the clusters and publish dates
            of the articles in the archive

            returns: Dict[publish_date int, Tuple[article_id str]]
        """
        # if self.trained_batches is None:
        #     self.trained_batches = set()
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
                # if b not in self.trained_batches and
                if len(b) > 0
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
