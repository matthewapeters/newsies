"""
tests.classify_test
"""

import pytest

from newsies.targets import DOCUMENT, HEADLINE
from newsies.actions import LIST
from newsies.classify import (
    prompt_analysis,
    categorize_text,
    TARGET_MAP,
    news_section,
)

# pylint: disable=broad-exception-caught, unused-variable

target_class_test_data = [
    (
        "Who are the key players in the current rift between the US and Canada?",
        "ENTITY",
    ),
    ("Are there any new stories about actor Kevin Kline?", DOCUMENT),
    ("What are the common themes in all of the news stories today?", DOCUMENT),
    (
        "What is the concensus about the US President covered in all of the world news today?",
        DOCUMENT,
    ),
    ("list the first five headlines from the science section", HEADLINE),
    ("how many headlines in the oditties secion today?", HEADLINE),
]


@pytest.mark.parametrize("query, expected", target_class_test_data)
def test__target_class(query, expected):
    """
    prompt_analysis:
      - Analyze a prompt
    """
    analysis = prompt_analysis(query)
    try:
        assert analysis
        assert "target" in analysis, f"Expected 'target' in analysis, got {analysis}"
        assert (
            expected == analysis["target"]
        ), f"expected target to be {expected}, got {analysis['target']}"
    except Exception as e:
        assert False, "ERROR: {e} not expected.  Got {analysis}"


def test__list_science_headlines():
    """
    test__list_science_headlines
    """
    query = "list the headlines from each of the articles in today's science section"
    intent = prompt_analysis(query)
    assert intent["context"] == "NEW"
    assert intent["target"] == HEADLINE
    assert intent["categories"][0] == "science"
    assert intent["quantity"] == "ALL"
    assert intent["action"] == LIST

    query = (
        "what is the most common targets from the list of headlines in the "
        "last prompt, ordered by the number of stories"
    )
    intent = prompt_analysis(query)
    assert intent["context"] == "OLD"
    assert intent["target"] == HEADLINE
    # assert intent["categories"][0] == "science"
    assert intent["quantity"] == "ALL"
    assert intent["action"] == LIST


def test__sections():
    """test__sections"""
    query = "list the headlines from each of the articles in today's science section"

    target_classification = categorize_text(query, list(TARGET_MAP.keys()))[:1]
    assert len(target_classification) == 1

    categories = news_section(query)
    assert len(categories) == 1
    assert categories[0] == "science"

    query = (
        "what is the most common targets from the list of headlines "
        "in the last prompt, ordered by the number of stories"
    )
    categories = news_section(
        query,
    )[:3]
    assert len(categories) == 0


article_reference_test_data = [
    ("read the first artcile from the list", 1),
    ("read the second article from the list", 2),
    ("get the three hundred thirty third article", 333),
]


@pytest.mark.parametrize("query, expected", article_reference_test_data)
def test__classify_with_ordinal(query, expected):
    """
    test__classify_with_ordinal
    """
    categorization = prompt_analysis(query)
    assert (
        categorization["ordinal"] == expected
    ), f"expected {expected}, got {categorization}"
