"""
tests.get_news_test
"""

from typing import Dict

from newsies.ap_news import (
    get_latest_news,
    article_loader,
    article_summary,
)
from newsies.document_structures import Document
from newsies.chroma_client import CRMADB
from newsies.targets import SUMMARY


def test__news_loader():
    """
    test__get_news
    """

    headlines: Dict[str, Document] = get_latest_news()
    article_loader(headlines)


def test__news_summarizer():
    """
    test__get_news
    """
    headlines: Dict[str, Document] = get_latest_news()
    article_summary(headlines)
    results = CRMADB.collection.get(where={"target": {"$eq": SUMMARY}})
    assert results
    assert len(results["ids"]) > 0
