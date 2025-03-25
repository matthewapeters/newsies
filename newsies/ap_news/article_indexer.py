"""
newsies.ap_news.article_index
"""

from multiprocessing import Pool
import pickle
from typing import Dict

from newsies.redis_client import REDIS
from newsies.document_structures import Document

from .index_visitor import IndexVisitor
from .article import Article


def index_article(
    story_url: str,
    task_status: dict = None,
    task_id: str = "",
    doc_id: int = 0,
    doc_count: int = 0,
):
    """
    index_article
        stores the article in ChromaDB vector space for clustering/searching
    """
    uri = REDIS.get(story_url)
    with open(uri, "rb") as fh:
        article: Article = pickle.load(fh)

    v = IndexVisitor()
    v.collection_name = "apnews.com"
    v.visit(article)

    article.pickle()
    if task_status is not None:
        task_status[task_id] = f"running: Indexing {doc_id} of {doc_count}"


def article_indexer(
    documents: Dict[str, Document] = None, task_state: dict = None, task_id: str = ""
):
    """
    article_indexer
    """

    if documents is None:
        pikl_path = "./daily_news/apnews.com/latest_news.pkl"
        with open(pikl_path, "rb") as fh:
            documents = pickle.load(fh)

    docs_count = len(documents)
    with Pool(processes=4) as ppool:
        ppool.starmap(
            index_article,
            [
                (
                    v.url,
                    task_state,  # task_status
                    task_id,  # task_id
                    i,  # doc_id
                    docs_count,  # doc_count
                )
                for i, v in enumerate(documents.values())
            ],
        )
