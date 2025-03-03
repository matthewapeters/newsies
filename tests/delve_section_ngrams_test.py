"""
tests.delve_section_ngrams_test
"""

import pytest

from newsies.chromadb_client import ChromaDBClient
from newsies.ap_news import SECTIONS
from newsies.ap_news.delve_section_ngrams import (
    extract_ngrams,
    get_section_doc_counts,
    # store_keywords_in_chromadb,
)

test_data = [
    (
        (
            "./tests/data/wyoming-abortion-clinic-wellspring-"
            "law-ban-37799481217b925085a812240850510c.txt"
        ),
        "",
    ),
]


@pytest.mark.parametrize("uri, expected", test_data)
def test__store_keywords_in_chromadb(uri, expected):
    """
    test__store_keywords_in_chromadb
    """
    crmadb = ChromaDBClient()
    crmadb.collection_name = "tests"
    with open(uri, "r", encoding="utf8") as fh:
        text = fh.read()

    ngram_counts = extract_ngrams(text, 30)
    assert ngram_counts
    assert ngram_counts == expected

    # store_keywords_in_chromadb(story=text, section="us-news", client=crmadb, n=30)


def test__get_section_doc_counts():
    """
    test__get_section_doc_counts
    """
    counts = get_section_doc_counts()
    assert (
        len(counts) == len(SECTIONS) - 1
    ), f"expected number of kesy to be {len(SECTIONS)-1}, got {len(counts)}"
    for k, v in counts.items():
        assert v > 0, f"expected len({k}) > 0"
