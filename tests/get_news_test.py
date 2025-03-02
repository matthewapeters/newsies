"""
tests.get_news_test
"""

from typing import Dict

from newsies.ap_news import (
    get_latest_news,
    news_loader,
    batch_news_summarizer,
)
from newsies.document_structures import Document
from newsies.chroma_client import CRMADB
from newsies.targets import SUMMARY


def test__news_loader():
    """
    test__get_news
    """

    headlines: Dict[str, Document] = get_latest_news()
    news_loader(headlines)


def test__news_summarizer():
    """
    test__get_news
    """
    headlines: Dict[str, Document] = get_latest_news()
    batch_news_summarizer(headlines)
    results = CRMADB.collection.get(where={"target": {"$eq": SUMMARY}})
    assert results
    assert len(results["ids"]) > 0
