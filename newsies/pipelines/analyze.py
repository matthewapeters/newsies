"""
newsies.pipelines.analyze
"""

from datetime import datetime

from newsies.ap_news.archive import Archive, get_archive
from newsies.llm.summarize_visitor import SummarizeVisitor

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

        # get the Archive
        archive: Archive = get_archive()

        # NOTE: Summaries, Questions, and Answers will be stored as both decoded
        # text and embedding in each article.  Creating the dataframes for training
        # that uses these will be part of the training pipeline.

        # Using a visitor, generate summaries for the articles in the Archive
        # and generate questions and answers from the article summaries. Add
        # summaries, questions, and answers to the articles
        v = SummarizeVisitor()
        v.visit(archive)

        # Using a visitor and Mistral-Instruct LLM, generate questions and answers
        # for NERs in the Archive articles.  Add questions and answers to the
        # articles in the Archive

        # Using a visitor and Mistral-Instruct LLM, generate questions and answers
        # for pairs of NERs in the Archive Articles.  Add questions and answers to the
        # articles in the Archive

        # OLD STUFF TO RELOCATE / RE-EVALUATE
        # print("\n\t- creating knn graph of stories\n")
        # TASK_STATUS[task_id] = "running - step: Creating KNN Graph"
        # generate_knn_graph()

        TASK_STATUS[task_id] = "complete"
    except Exception as e:
        TASK_STATUS[task_id] = f"error: {e}"
        raise e
