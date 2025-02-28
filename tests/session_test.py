"""
tests.session_test
"""

from newsies.session import Session
from newsies.llm import LLM as llm
from newsies.chromadb_client import ChromaDBClient


def test__list_all_science_headlines():
    """
    test__list_all_science_headlines
        test to get the list of all of a day's science headlines
        and then analyze them for common themes
    """
    db = ChromaDBClient()
    db.collection_name = "ap_news_2025-02-28"
    s = Session(llm, db)
    query = "list the headlines from each of the articles in today's science section"
    response = s.query(query)
    assert response
    query = (
        "what is the most common theme from the list of headlines in the last prompt"
    )
