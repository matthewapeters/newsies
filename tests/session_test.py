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
    db.collection_name = "ap_news_2025-03-01"
    s = Session(llm, db)
    query = (
        "list only the headlines from each of the articles in today's science section."
    )
    response = s.query(query)
    assert response
    query = (
        "given the list of headlines from the last prompt, "
        "determine up to three common themes and list each headline under its theme."
        "Do not explain your thinking. only respond with JSON. "
        'Example: {"brief theme description": '
        '["a headline related to this theme", "another headline related to this theme"]}'
    )
    response = s.query(query)
    assert response

    query = "read the second article from the previous query."
    response = s.query(query)
    assert response

    query = "read the summary of the third article from the previous query."
    response = s.query(query)
    assert response


if __name__ == "__main__":
    test__list_all_science_headlines()
