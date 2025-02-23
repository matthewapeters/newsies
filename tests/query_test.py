import pytest

from newsies.classify import prompt_analysis
from newsies.llm import LLM as llm
from newsies.chroma_client import CRMADB

query_data = [("any stories about science today?", "")]
collection = f"ap_news_2025-02-21"
CRMADB.collection = collection


@pytest.mark.parametrize("query, expected", query_data)
def test__read_news(query, expected):
    # analyze request
    request_meta = prompt_analysis(query)
    print(f"(newsies thinks you want to know about {request_meta})")
    # Generate response using GPT4All
    response = CRMADB.generate_rag_response(query.lower(), request_meta, llm, 5)
    print("\nRESPONSE:\n", response)
