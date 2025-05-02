"""
newsies.pipelines.analyze
"""

from datetime import datetime

from newsies.ap_news.knn_analysis import generate_knn_graph

from .task_status import TASK_STATUS


# pylint: disable=broad-exception-caught


def analyze_pipeline(task_id: str, archive: str = None):
    """
    analyze_pipeline
    """
    print("\nANALYZE NEWS\n")
    if archive is None:
        archive = datetime.now().strftime(r"%Y-%m-%d")
    TASK_STATUS[task_id] = "start"
    try:

        print("\n\t- creating knn graph of stories\n")
        TASK_STATUS[task_id] = "running - step: Creating KNN Graph"
        generate_knn_graph()

        TASK_STATUS[task_id] = "complete"
    except Exception as e:
        TASK_STATUS[task_id] = f"error: {e}"
        raise e
