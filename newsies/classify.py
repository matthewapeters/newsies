"""
newsies.classify
"""

from typing import List
import torch
from transformers import pipeline

from newsies.chromadb_client import find_ordinal, ChromaDBClient
from newsies.classification_heuristics import (
    ACTION_HEURISTICS,
    TARGET_HEURISTICS,
    ONE_MANY_ALL_HEURISTICS,
    NEWS_SECTION_HEURISTICS,
    QUERY_OR_REFERENCE_HEURISTICS,
)

# pylint: disable=global-statement, fixme

TAGS_COLLECTION = "newsies_tags"
tags_db = ChromaDBClient()
tags_db.collection_name = TAGS_COLLECTION
tags_db.language = "en"


DEVICE_STR = (
    # f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
categorizer = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli", device=DEVICE_STR
)


def embed_targets(target_map: dict):
    """
    embed_targets
     - upsert targets as embeddings to ChromaDB
    """
    global TARGET_MAP
    TARGET_MAP = target_map
    # tags_db.embed_documents(
    #    document_ids=[k.replace(" ", "_") for k in target_map.keys()],
    #    docs=[k for k in target_map.keys()],
    #    metadata=[{"target": v} for v in target_map.values()],
    # )


# Ensure the tags database has at least the default tags
embed_targets(_default_targets)


def refresh_targets():
    """
    refresh_targets
     - load all of the targets from the ChromaDB collection

     These can be updated over time to identify preferred classifications.  As these
     are used in prompts analysis, they can customize retrieval based on user idioms
    """
    _all_targets = tags_db.collection.get()
    _target_count = len(_all_targets["documents"])
    global TARGET_MAP

    TARGET_MAP = {
        _all_targets["documents"][i]: _all_targets["metadatas"][i]["target"]
        for i in range(_target_count)
    }


# TODO - uncomment when we want to retrieve the mappings from the chromadb
# refresh_targets()


def categorize_text(text, labels: List[List[str]]):
    """
    categorize_text
        Function to classify text using zero-shot classification
    """
    result = categorizer(text, labels, multi_label=False)
    return [(k, v) for k, v in dict(zip(result["labels"], result["scores"])).items()]


def prompt_analysis(query) -> str:
    """
    prompt_analysis:
      - Analyze a prompt
    """
    ordinal_dict = find_ordinal(query) or {"number": None}

    label_sets = [
        list(QUERY_OR_REFERENCE_HEURISTICS.keys()),
        list(TARGET_HEURISTICS.keys()),
        list(NEWS_SECTION_HEURISTICS.keys()),
        list(ONE_MANY_ALL_HEURISTICS.keys()),
        list(ACTION_HEURISTICS.keys()),
    ]

    classified_results = categorize_text(query, label_sets)
    return {
        "context": QUERY_OR_REFERENCE_HEURISTICS[classified_results[label_sets[0]]],
        "target": TARGET_HEURISTICS[classified_results[label_sets[1]]],
        "section": NEWS_SECTION_HEURISTICS[classified_results[label_sets[2]]],
        "quantity": ONE_MANY_ALL_HEURISTICS[classified_results[label_sets[3]]],
        "action": ACTION_HEURISTICS[classified_results[label_sets[4]]],
        "ordinal": [ordinal_dict["number"], ordinal_dict["distance"]],
    }
