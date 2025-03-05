"""
newsies.pipelines.analyze
"""

from newsies.ap_news import (
    batch_news_summarizer,
    analyze_ngrams_per_section,
    compute_tfidf,
)
from .task_status import TASK_STATUS


def analyze_pipeline(task_id: str):
    """
    analyze_pipeline
    """
    print("\nANALYZE NEWS\n")
    # print("\t- retrieving headlines\n")
    # from newsies.ap_news.latest_news import get_latest_news
    # headlines: Dict[str, Document] = get_latest_news()
    TASK_STATUS[task_id] = "start"
    try:
        print("\n\t- summarizing stories")
        TASK_STATUS[task_id] = "running - step: summarizing stories"
        batch_news_summarizer()

        print("\n\t- detecting ngrams specific to news sections\n")
        TASK_STATUS[task_id] = "running - step: extracing named entities and n-grams"
        analyze_ngrams_per_section()

        print("\n\t- computing tf-idf for ngrams\n")
        TASK_STATUS[task_id] = (
            "running - step: computing tf-idf for named entities and n-grams"
        )
        compute_tfidf()
        TASK_STATUS[task_id] = "complete"
    except Exception as e:
        TASK_STATUS[task_id] = f"error: {e}"
