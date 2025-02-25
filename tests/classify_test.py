import json
import pytest


from newsies.classify import prompt_analysis, categorize_text, THEME_MAP

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


def test__list_science_headlines():
    query = "list the headlines from each of the articles in today's science section"
    intent = prompt_analysis(query)
    assert intent["context"] == "NEW"
    assert intent["theme"] == "DOCUMENT"
    assert intent["categories"][0] == "science"
    assert intent["quantity"] == "ALL"
    assert intent["action"] == "LIST-HEADLINE"

    query = "what is the most common themes from the list of headlines in the last prompt, ordered by the number of stories"
    intent = prompt_analysis(query)
    assert intent["context"] == "OLD"
    assert intent["theme"] == "DOCUMENT"
    # assert intent["categories"][0] == "science"
    assert intent["quantity"] == "ALL"
    assert intent["action"] == "LIST-HEADLINE"


def test_categories():

    category_list = [
        "world-news",
        "us-news",
        "politics",
        "business",
        "technology",
        "science",
        "health",
        "entertainment",
        "sports",
        "oddities",
    ]
    query = "list the headlines from each of the articles in today's science section"

    theme_classification = categorize_text(
        query, list(THEME_MAP.keys()), threshold=None
    )[:1]
    assert len(theme_classification) == 1

    categories = categorize_text(
        query,
        category_list,
    )[:3]
    assert len(categories) == 1
    assert categories[0] == "science"

    query = "what is the most common themes from the list of headlines in the last prompt, ordered by the number of stories"
    categories = categorize_text(
        query,
        category_list,
    )[:3]
    assert len(categories) == 0
