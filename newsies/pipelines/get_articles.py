"""
newsies.pipelines.get_articles
"""

# pylint: disable=broad-exception-caught

from newsies.ap_news.latest_news import (
    get_latest_news,
)
from newsies.ap_news.article_ner import article_ner
from newsies.ap_news.article_embeddings import article_embeddings

# from newsies.ap_news.article_summary import article_summary
from newsies.ap_news.article_loader import article_loader
from newsies.ap_news.article_indexer import article_indexer

from .task_status import TASK_STATUS


def get_articles_pipeline(task_id: str):
    """
    get_articles_pipeline
    """
    print("\nGET ARTICLES\n")
    TASK_STATUS[task_id] = "started"
    try:
        print("\n\t- retrieving headlines\n")
        TASK_STATUS[task_id] = "running - step: retrieving headlines"
        # get the latest news links from AP press and pickle for downstream use
        get_latest_news()

        print("\n\t- article loader\n")
        TASK_STATUS[task_id] = "running - step: retrieving and caching news articles"
        article_loader(task_state=TASK_STATUS, task_id=task_id)

        print("\n\t- article NER\n")
        TASK_STATUS[task_id] = "running - step: detecting named entities in articles"
        article_ner(task_state=TASK_STATUS, task_id=task_id)

        # summaries may not be needed until after model training
        # print("\n\t- article summary\n")
        # TASK_STATUS[task_id] = "running - step: generating article summaries"
        # article_summary(task_state=TASK_STATUS, task_id=task_id)

        print("\n\t- article embeddings\n")
        TASK_STATUS[task_id] = "running - step: generating embeddings"
        article_embeddings(task_state=TASK_STATUS, task_id=task_id)

        print("\n\t- article indexing\n")
        TASK_STATUS[task_id] = "running - step: indexing articles"
        article_indexer(task_state=TASK_STATUS, task_id=task_id)

        TASK_STATUS[task_id] = "complete"
    except Exception as e:
        TASK_STATUS[task_id] = f"error: {e}"
