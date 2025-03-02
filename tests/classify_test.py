"""
tests.classify_test
"""

from typing import Dict
import pytest


from newsies.targets import DOCUMENT, HEADLINE, SUMMARY, ENTITY
from newsies.actions import LIST
from newsies.classify import (
    prompt_analysis,
    categorize_text,
    #    TARGET_HEURISTICS,
    NEWS_SECTION_HEURISTICS,
)

# pylint: disable=broad-exception-caught, unused-variable

target_class_test_data = [
    (
        "Who are the key players in the current rift between the US and Canada?",
        ENTITY,
    ),
    ("Are there any new stories about actor Kevin Kline?", DOCUMENT),
    ("What are the common themes in all of the news stories today?", DOCUMENT),
    (
        "What is the concensus about the US President covered in all of the world news today?",
        DOCUMENT,
    ),
    ("list the first five headlines from the science section", HEADLINE),
    ("how many headlines in the oditties secion today?", HEADLINE),
    ("get the summaries of each of last three articles", SUMMARY),
    ("read the summary of the first story in the list", SUMMARY),
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
        assert (
            "target" in analysis
        ), f"Expected 'target' in analysis, got {list(analysis.keys())}"
        assert (
            expected == analysis["target"]
        ), f"expected target to be {expected}, got {analysis['target']}"
    except Exception as e:
        assert False, f"ERROR: {e} not expected.  Got {analysis}"


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


news_sections_testdata = [
    (
        "list the headlines from each of the articles in today's science section",
        "science",
    ),
    (
        "list the headlines from each of the articles in today's us news section",
        "us-news",
    ),
    ("any news from Africa today?", "world-news"),
    ("any news from Canada today?", "world-news"),
    ("any news from Asia today?", "world-news"),
    ("any robotics news today?", "technology"),
    ("what  is the president up to today?", "politics"),
    ("did that bill pass in the sentate?", "politics"),
    ("any new scandals in the congress?", "politics"),
]


@pytest.mark.parametrize("query,section", news_sections_testdata)
def test__news_sections(query, section):
    """test__news_sections"""
    keys = list(NEWS_SECTION_HEURISTICS.keys())

    section_class: Dict[str, float] = categorize_text(query, [keys])
    assert section == NEWS_SECTION_HEURISTICS[section_class[keys[0]]], (
        f"expected {section}, got {section_class[keys[0]]} -> "
        f"{NEWS_SECTION_HEURISTICS[section_class[keys[0]]]}"
    )


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
