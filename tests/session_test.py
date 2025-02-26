from newsies.session import Session
from newsies.llm import LLM as llm
from newsies.chromadb_client import ChromaDBClient


def test__list_all_science_headlines():
    db = ChromaDBClient()
    db.collection = "ap_news_2025-02-25"
    s = Session(llm, db)
    query = "list the headlines from each of the articles in today's science section"
    response = s.query(query)
    query = (
        "what is the most common theme from the list of headlines in the last prompt"
    )
