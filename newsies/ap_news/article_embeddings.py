"""newsies.ap_news.article_embeddings"""

from multiprocessing import Pool
import pickle
from typing import Dict

from newsies.document_structures import Document
from newsies.redis_client import REDIS

from .embedding_visitor import EmbeddingVisitor
from .article import Article


def generate_embeddings(
    story_url: str,
    task_status: dict = None,
    task_id: str = "",
    doc_id: int = 0,
    doc_count: int = 0,
):
    """
    generate_embeddings
    """
    uri = REDIS.get(story_url)
    with open(uri, "rb") as fh:
        article: Article = pickle.load(fh)

    v = EmbeddingVisitor()
    v.visit(article)
    article.pickle()
    if task_status is not None:
        task_status[task_id] = f"running: embeddings {doc_id} of {doc_count}"


def article_embeddings(
    documents: Dict[str, Document] = None, task_state: dict = None, task_id: str = ""
):  # pylint: disable=unused-argument
    """
    article_embeddings
    """
    if documents is None:
        pikl_path = "./daily_news/apnews.com/latest_news.pkl"
        with open(pikl_path, "rb") as fh:
            documents = pickle.load(fh)

    with Pool(processes=4) as ppool:
        ppool.starmap(
            generate_embeddings,
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
