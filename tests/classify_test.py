import json
import pytest


# from newsies.main import CRMADB
from newsies.ap_news.main import PAGES
from newsies.classify import (
    prompt_analysis,
)

prompt_test_data = [
    (
        "Who are the key players in the current rift between the US and Canada?",
        {"theme": "ENTITY", "category": "us-news"},
    ),
    (
        "Are there any new stories about actor Kevin Kline?",
        {"theme": "DOCUMENT", "category": "entertainment"},
    ),
    (
        "What are the common themes in all of the news stories today?",
        {"theme": "SUMMARY", "category": "world-news"},
    ),
    (
        "What is the concensus about the US President as reported in all of the world news stories?",
        {"theme": "SUMMARY", "category": "world-news"},
    ),
]


@pytest.mark.parametrize("query, expected", prompt_test_data)
def test__classify_prompt_analysis(query, expected):
    """
    prompt_analysis:
      - Analyze a prompt
    """
    result = prompt_analysis(query)
    assert result
    try:
        analysis = json.loads(result)
        assert len(analysis) == 2, f"Expected 2 items, got {len(analysis)}"
        assert "categories" in analysis, f"Expected 'categories' in analysis"
        assert "theme" in analysis, f"Expected 'theme' in analysis"
        assert isinstance(
            analysis["categories"], list
        ), f"Expected list, got {type(analysis['categories'])}"
        assert isinstance(
            analysis["theme"], str
        ), f"Expected string, got {type(analysis['theme'])}"
        assert (
            analysis["theme"] == expected["theme"]
        ), f"Expected {expected['theme']} got {analysis['theme']}"
        assert (
            expected["category"] in analysis["categories"]
        ), f"expected {expected['category']} to be in categories, but only found {analysis['categories']}"
    except Exception as e:
        assert False, f"Error: {e}\t{result}"
