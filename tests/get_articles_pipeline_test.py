"""
tests.get_articles_pipeline_test
"""

import pickle
from typing import Dict

from newsies.pipelines.get_articles import get_articles_pipeline
from newsies.pipelines.task_status import TASK_STATUS
from newsies.redis_client import REDIS
from newsies.ap_news.article import Article


def test__get_articles_pipeline():
    """
    test__get_articles_pipeline
    """
    task_id = "test_task_id"
    TASK_STATUS[task_id] = {
        "session_id": "N/A",
        "status": "queued",
        "task": "get-articles",
        "username": "tester",
    }
    get_articles_pipeline(task_id)
    assert TASK_STATUS[task_id]["status"] == "complete"

    documents: Dict[str, Dict[str, str]] = {}
    pikl_path = "./daily_news/apnews.com/latest_news.pkl"
    with open(pikl_path, "rb") as fh:
        documents = pickle.load(fh)

    assert len(documents) > 0
    for d in documents:
        uri = REDIS.get(d)
        assert uri is not None
        try:
            with open(uri, "rb") as fh:
                article: Article = pickle.load(fh)
                assert article is not None
                assert len(article.story) > 0
                # assert len(article.summary) > 0 and len(article.summary) < len(
                #    article.story
                # )
                assert len(article.keywords) > 0
                assert len(article.named_entities) > 0
                assert len(article.embeddings) > 0
                assert len(article.pipelines) >= 3
        except FileNotFoundError:
            pass
