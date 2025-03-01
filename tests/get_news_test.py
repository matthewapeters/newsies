"""
tests.get_news_test
"""

from typing import Dict

from newsies.ap_news import (
    get_latest_news,
    news_loader,
    news_summarizer,
)
from newsies.document_structures import Document


def test__get_news():
    """
    test__get_news
    """

    headlines: Dict[str, Document] = get_latest_news()
    news_loader(headlines)
    news_summarizer(headlines)
