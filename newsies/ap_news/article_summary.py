"""
newsies.ap_news.article_summary
"""

from multiprocessing import Pool
import pickle
from typing import Dict

import torch

from newsies.redis_client import REDIS
from newsies.document_structures import Document

from .article import Article
from .summary_visistor import SummaryVisitor


def generate_summary(
    story_url: str,
    task_status: dict = None,
    task_id: str = "",
    doc_id: int = 0,
    doc_count: int = 0,
):
    """
    generate_summary
    """
    uri = REDIS.get(story_url)
    with open(uri, "rb") as fh:
        article: Article = pickle.load(fh)

    v = SummaryVisitor()
    v.visit(article)
    article.pickle()
    if task_status is not None:
        task_status[task_id] = f"running: summary {doc_id} of {doc_count}"


def article_summary(
    documents: Dict[str, Document] = None, task_state: dict = None, task_id: str = ""
):
    """
    article_summary
    """
    if documents is None:
        pikl_path = "./daily_news/apnews.com/latest_news.pkl"
        with open(pikl_path, "rb") as fh:
            documents = pickle.load(fh)

    with Pool(processes=4) as ppool:
        ppool.starmap(
            generate_summary,
            [
                (
                    v.url,
                    task_state,  # task_status
                    task_id,  # task_id
                    i,  # doc_id
                    len(documents),  # doc_count
                )
                for i, v in enumerate(documents.values())
            ],
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
