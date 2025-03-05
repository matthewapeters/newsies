"""
tests.delve_section_ngrams_test
"""

import json

import pytest


from newsies.chromadb_client import ChromaDBClient
from newsies.ap_news import SECTIONS
from newsies.ap_news.delve_section_ngrams import (
    extract_ngrams,
    get_section_doc_counts,
    store_keywords_in_chromadb,
)
from newsies.collections import TAGS

test_data = [
    (
        (
            "./tests/data/wyoming-abortion-clinic-wellspring-"
            "law-ban-37799481217b925085a812240850510c.txt"
        ),
        87,
    ),
]


@pytest.mark.parametrize("uri, expected", test_data)
def test__store_keywords_in_chromadb(uri, expected):
    """
    test__store_keywords_in_chromadb
    """
    with open(uri, "r", encoding="utf8") as fh:
        text = fh.read()

    ngram_counts = extract_ngrams(text, 30)
    assert ngram_counts
    common_over_two = [n for n in ngram_counts.most_common() if n[1] >= 2]
    assert (
        len(common_over_two) == expected
    ), f"expected most_common with count>2 to be {expected}, got {len(common_over_two)}"

    sections = ["politics", "us-news"]
    client = ChromaDBClient()
    client.collection_name = TAGS
    store_keywords_in_chromadb(
        chroma_client=client, stories=[text], sections=[sections]
    )
    doc_id = f"ngram_{common_over_two[0][0]}"
    verify = client.collection.get(ids=[doc_id])
    v_sections = json.loads(verify["metadatas"][0]["sections"])
    for s in sections:
        assert (
            s in v_sections
        ), f"expected {s} to be in returned sections, got {list(v_sections.keys())}"


def test__get_section_doc_counts():
    """
    test__get_section_doc_counts
    """
    counts = get_section_doc_counts()
    assert (
        len(counts) == len(SECTIONS) - 1
    ), f"expected number of keys to be {len(SECTIONS)-1}, got {len(counts)}"
    for k, v in counts.items():
        assert v > 0, f"expected len({k}) > 0"


@pytest.mark.parametrize("uri, expected", test_data)
def test__generate_ngrams(uri, expected):
    """
    test__generate_ngrams
      - this test will help determine if we can improve quality of ngrams
      with nltk or spaCy through named entity recognition (NER) and removal
      of stopwords
    """

    with open(uri, "r", encoding="utf8") as fh:
        text = fh.read()
