"""
newsies.pipelines.get_news
"""

from newsies.ap_news.latest_news import (
    get_latest_news,
    news_loader,
    headline_loader,
)
from .task_status import TASK_STATUS

# pylint: disable=broad-exception-caught


def get_news_pipeline(task_id: str):
    """
    get_news_pipeline
    """
    print("\nGET NEWS\n")
    TASK_STATUS[task_id] = "start"
    try:
        print("\n\t- retrieving headlines\n")
        TASK_STATUS[task_id] = "running - step: retrieving headlines"
        # get the latest news links from AP press and pickle for downstream use
        get_latest_news()

        print("\n\t- news loader\n")
        TASK_STATUS[task_id] = "running - step: retrieving and caching news stories"
        news_loader()

        print("\n\t- headlines loader\n")
        TASK_STATUS[task_id] = "running - step: loading headlines"
        headline_loader()
        TASK_STATUS[task_id] = "complete"
    except Exception as e:
        TASK_STATUS[task_id] = f"error: {e}"
