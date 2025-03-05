"""
newsies.pipelines.get_news
"""

from newsies.ap_news.latest_news import (
    get_latest_news,
    news_loader,
    headline_loader,
)


def get_news_pipeline():
    """
    get_news_pipeline
    """
    print("\nGET NEWS\n")
    print("\n\t- retrieving headlines\n")
    # get the latest news links from AP press and pickle for downstream use
    get_latest_news()

    print("\n\t- news loader\n")
    news_loader()

    print("\n\t- headlines loader\n")
    headline_loader()
