"""
newsies.classify
"""

import torch
from transformers import pipeline

from newsies.chromadb_client import find_ordinal, ChromaDBClient

TAGS_COLLECTION = "newsies_tags"
tags_db = ChromaDBClient()
tags_db.collection_name = TAGS_COLLECTION
tags_db.language = "en"


device_str = (
    # f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"
    f"cuda"
    if torch.cuda.is_available()
    else "cpu"
)
categorizer = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli", device=device_str
)


target_MAP = {}
_default_targets = {
    "detailed news story": "DOCUMENT",
    "read an article": "DOCUMENT",
    "any news stories": "DOCUMENT",
    "statistical summary of multiple news stories": "DOCUMENT",
    "common narrative targets in multiple stories": "DOCUMENT",
    "list headlines about a common person, place, or thing": "HEADLINE",
    "list headlines from all stories or from a category of stories": "HEADLINE",
    "common targets in multiple headlines": "HEADLINE",
    "staticical summary of multiple headlines": "HEADLINE",
    "people, places, things in news stories": "ENTITY",
    "people, places, things in headlines": "ENTITY",
}


def embed_targets(target_map: dict):
    """
    embed_targets
     - upsert targets as embeddings to ChromaDB
    """
    tags_db.embed_documents(
        document_ids=[k.replace(" ", "_") for k in target_map.keys()],
        docs=[k for k in target_map.keys()],
        metadata=[{"target": v} for v in target_map.values()],
    )


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
    global target_MAP

    target_MAP = {
        _all_targets["documents"][i]: _all_targets["metadatas"][i]["target"]
        for i in range(_target_count)
    }


refresh_targets()


# Function to classify text using zero-shot classification
def categorize_text(text, labels):
    result = categorizer(text, labels, multi_label=False)
    return [(k, v) for k, v in dict(zip(result["labels"], result["scores"])).items()]


def determine_quantities(query):
    heuristics = {
        "referring to one story or article, story, person, place or thing": "ONE",
        "referring to more than one, but less than all articles, stories, people, places or things": "MANY",
        "referring to all articles, stories, people, places or things": "ALL",
        "request is for the full set of articles, stories, people, places or things from a category": "ALL",
        "request if for each document from one or more categories": "ALL",
    }
    classification = categorize_text(query, list(heuristics.keys()))[0][0]
    print(f"\nQUANTITIES: {classification}\n")
    return heuristics[classification]


def new_or_old_query(query):
    heuristics = {
        "refers to 'today'": "NEW",
        "refers to 'on this day'": "NEW",
        "refers to new query": "NEW",
        "introduces a change of topic": "NEW",
        "refers to prior query or prompt": "OLD",
        "refers somthing we talked about before": "OLD",
    }
    classification = categorize_text(query, list(heuristics.keys()))[0][0]
    print(f"\nNEW OR OLD: {classification}\n")
    return heuristics[classification]


def determine_action(query):
    heuristics = {
        "read an article": "READ",
        "list one or more titles": "LIST-HEADLINE",
        "list one or more headlines": "LIST-HEADLINE",
        "enumerate or list stories": "LIST-HEADLINE",
        "count articles": "COUNT",
        "summarize multiple articles together": "SYNTHESIZE",
        "summarize a single article": "SUMMARIZE",
    }
    classification = categorize_text(query, list(heuristics.keys()))[0][0]
    print(f"\nACTION: {classification}\n")
    return heuristics[classification]


def news_categories(query) -> str:
    heuristics = {
        "global news": "world-news",
        "world news": "world-news",
        "international news": "world-news",
        "us news": "us-news",
        "USA": "us-news",
        "domestic": "us-news",
        "state": "us-news",
        "president": "politics",
        "congress": "politics",
        "supreme court": "politics",
        "senate": "politics",
        "stock market": "business",
        "corporate": "business",
        "business": "business",
        "technology": "technology",
        "science": "technology",
        "space": "technology",
        "nasa": "technology",
        "robotics": "science",
        "physics": "science",
        "biology": "science",
        "research": "science",
        "health": "health",
        "medical": "health",
        "health": "health",
        "entertainment": "entertainment",
        "actor": "entertainment",
        "film": "entertainment",
        "broadway": "entertainment",
        "sports": "sports",
        "football": "sports",
        "nfl": "sports",
        "nba": "sports",
        "varsity": "sports",
        "basketball": "sports",
        "baseball": "sports",
        "olympics": "sports",
        "athelete": "sports",
        "league": "sports",
        "oddities": "oddities",
        "unusual": "oddities",
        "quirky": "oddities",
        "strange": "oddities",
    }
    classification = categorize_text(
        query,
        list(heuristics.keys()),
    )
    print(f"\nCLASSIFICATION: {classification}\n")
    return [heuristics[c] for c, _ in classification][:3]


def target_class(query) -> str:
    target_class = categorize_text(query, list(target_MAP.keys()))[0]
    return target_MAP[target_class]


def prompt_analysis(query) -> str:
    """
    prompt_analysis:
      - Analyze a prompt
    """
    target = target_class(query)
    categories = news_categories(query)
    quantity = determine_quantities(query)
    new_or_old = new_or_old_query(query)
    action = determine_action(query)
    ordinal_dict = find_ordinal(query) or {"number": None}

    return {
        "context": new_or_old,
        "target": target,
        "categories": categories,
        "quantity": quantity,
        "ordinal": [ordinal_dict["number"], ordinal_dict["distance"]],
        "action": action,
    }
