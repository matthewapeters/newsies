"""
tests.ngram_section_detect_test
"""

from typing import List
import pytest
from newsies.classify.ngram_section_detect import get_relevant_sections


test_data = [
    ("are there any stories about Wyoming", ["us-news"]),
    ("are there any stories about Blue Origin", ["science"]),
    ("I want the most recent story about the Oscar awards", ["entertainment"]),
    ("any updates on Canada's response to US tariffs?", ["business", "world-news"]),
    ("any news about robotics or AI?", ["technology"]),
    (
        "are there any actors appearing in more than one movie right now?",
        ["entertainment"],
    ),
    (
        "what are reactions to JD Vance's white house debacle?",
        ["us-news", "politics"],
    ),
    (
        "doge downsizing of food and drug administration",
        ["health"],
    ),
    ("what are stores doing to battle inflation", ["business"]),
]


@pytest.mark.parametrize("prompt, sections", test_data)
def test__ngram_section_detect(prompt: str, sections: List[str]):
    """
    test__ngram_section_detect
    """
    best_fit: List[str] = get_relevant_sections(prompt)
    if len(sections) == 1:
        assert (
            sections[0] == best_fit[0]
        ), f"expected {sections[0]} to be the best fit but got {best_fit[0]}"
    else:
        for section in sections:
            assert section in best_fit, f"expected {section} to be in {best_fit}"
