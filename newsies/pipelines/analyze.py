"""
newsies.pipelines.analyze
"""

from datetime import datetime

from newsies.ap_news import (
    batch_news_summarizer,
    analyze_ngrams_per_section,
    compute_tfidf,
)
from .task_status import TASK_STATUS

# pylint: disable=broad-exception-caught


def analyze_pipeline(task_id: str, archive: str = None):
    """
    analyze_pipeline
    """
    print("\nANALYZE NEWS\n")
    # print("\t- retrieving headlines\n")
    # from newsies.ap_news.latest_news import get_latest_news
    # headlines: Dict[str, Document] = get_latest_news()
    if archive is None:
        archive = datetime.now().strftime(r"%Y-%m-%d")
    TASK_STATUS[task_id] = "start"
    try:
        print("\n\t- summarizing stories")
        TASK_STATUS[task_id] = "running - step: summarizing stories"
        batch_news_summarizer(archive=archive)

        print("\n\t- detecting ngrams specific to news sections\n")
        TASK_STATUS[task_id] = "running - step: extracing named entities and n-grams"
        analyze_ngrams_per_section(archive=archive)

        print("\n\t- computing tf-idf for ngrams\n")
        TASK_STATUS[task_id] = (
            "running - step: computing tf-idf for named entities and n-grams"
        )
        compute_tfidf()
        TASK_STATUS[task_id] = "complete"
    except Exception as e:
        TASK_STATUS[task_id] = f"error: {e}"
