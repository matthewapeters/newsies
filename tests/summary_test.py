"""
tests.summary_test
"""

import pytest

# from newsies.summarizer import summarize_story
from newsies.ap_news import summarize_story
from newsies.chroma_client import CRMADB

summary_test_data = [
    (
        {
            "uri": (
                "./tests/data/wind-energy-colombia-wayuu-indigenous-"
                "resistance-clash-cemetery-renewable-e55077418352f19349dc27b09f1eee18.txt"
            )
        },
        {"length": 1024},
    ),
    (
        {
            "uri": (
                "./tests/data/tv-procedurals-watson-doctor-odyssey-high-potential"
                "-elsbeth-matlock-77d1502b193df03c5cbebc99fe991934.txt"
            )
        },
        {"length": 1024},
    ),
]


@pytest.mark.parametrize("inputs, expected", summary_test_data)
def test__story_summarizer(inputs, expected):
    """
    test__story_summarizer
    """
    summary = summarize_story(inputs["uri"], CRMADB)
    assert summary
    assert (
        len(summary) <= expected["length"]
    ), f"expected summary length to be less than {expected['length']}, got {len(summary)}"
