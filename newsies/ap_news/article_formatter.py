"""
newsies.ap_news.article_index
"""

from multiprocessing import Pool
import pickle
from typing import Dict

from newsies.redis_client import REDIS
from newsies.document_structures import Document

from .article_format_visitor import ArticleFormatVisitor
from .article import Article


def format_article(
    story_url: str,
    task_status: dict = None,
    task_id: str = "",
    doc_id: int = 0,
    doc_count: int = 0,
):
    """
    format_article
        stores the article in ChromaDB vector space for clustering/searching
    """
    if story_url is None:
        return
    uri = REDIS.get(story_url)
    if uri is None:
        return
    with open(uri, "rb") as fh:
        article: Article = pickle.load(fh)

    v = ArticleFormatVisitor()
    v.visit(article)
    article.pickle()
    if task_status is not None:
        task_status[task_id] = f"running: Indexing {doc_id} of {doc_count}"


def article_formatter(
    documents: Dict[str, Document] = None, task_state: dict = None, task_id: str = ""
):
    """
    article_formatter
        generate a formatted string for the article with the following format:
        <item_id>
        <publish_date>
        <section_title>: <headline> ...
        <author>...
        <article>

        the format uses semantic tags that will be added to the LLM for training
    """

    if documents is None:
        pikl_path = "./daily_news/apnews.com/latest_news.pkl"
        with open(pikl_path, "rb") as fh:
            documents = pickle.load(fh)

    docs_count = len(documents)
    with Pool(processes=4) as ppool:
        ppool.starmap(
            format_article,
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
