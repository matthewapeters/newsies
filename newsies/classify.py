"""
newsies.classify
"""

from typing import Dict, List
import torch
from transformers import pipeline
from newsies.chromadb_client import find_ordinal
from newsies.classification_heuristics import (
    ACTION_HEURISTICS,
    TARGET_HEURISTICS,
    ONE_MANY_ALL_HEURISTICS,
    NEWS_SECTION_HEURISTICS,
    QUERY_OR_REFERENCE_HEURISTICS,
)


DEVICE_STR = (
    # f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
categorizer = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli", device=DEVICE_STR
)


def categorize_text(text, label_sets: List[str]) -> Dict[str, float]:
    """
    categorize_text:
        Classifies a single text against multiple label sets in one call.
    """
    results = categorizer(text, sum(label_sets, []), multi_label=False)

    # Group results back into original categories
    label_scores = dict(zip(results["labels"], results["scores"]))

    classified = {}
    for label_set in label_sets:
        best_label = max(label_set, key=lambda label: label_scores.get(label, 0))
        classified[label_set[0]] = best_label  # Assign category by first label set key

    return classified


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
        "context": QUERY_OR_REFERENCE_HEURISTICS[classified_results[label_sets[0][0]]],
        "target": TARGET_HEURISTICS[classified_results[label_sets[1][0]]],
        "section": NEWS_SECTION_HEURISTICS[classified_results[label_sets[2][0]]],
        "quantity": ONE_MANY_ALL_HEURISTICS[classified_results[label_sets[3][0]]],
        "action": ACTION_HEURISTICS[classified_results[label_sets[4][0]]],
        "ordinal": [ordinal_dict["number"], ordinal_dict["distance"]],
    }
