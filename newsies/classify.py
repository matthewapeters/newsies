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


_default_targets = {
    "story": "DOCUMENT",
    "stories": "DOCUMENT",
    "article": "DOCUMENT",
    "articles": "DOCUMENT",
    "details news story": "DOCUMENT",
    "read story": "DOCUMENT",
    "read article": "DOCUMENT",
    "any news stories": "DOCUMENT",
    "headline": "HEADLINE",
    "title": "HEADLINE",
    "headlines": "HEADLINE",
    "titles": "HEADLINE",
    "list headlines": "HEADLINE",
    "list titles": "HEADLINE",
}
TARGET_MAP = {}


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
    pass


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
        "pull an article": "READ",
        "read a story": "READ",
        "pull a story": "READ",
        "list titles": "LIST",
        "list headlines": "LIST",
        "list stories": "LIST",
        "list articles": "LIST",
        "count articles": "COUNT",
        "summary of an article": "SUMMARY",
        "summary of a story": "SUMMARY",
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
        "robotics": "technology",
        "computers": "technology",
        "ai": "technology",
        "nasa": "science",
        "space": "science",
        "science": "science",
        "research": "science",
        "physics": "science",
        "biology": "science",
        "health": "health",
        "medical": "health",
        "health": "health",
        "well being": "",
        "entertainment": "entertainment",
        "actor": "entertainment",
        "oscars": "entertainment",
        "emmy": "entertainment",
        "cma": "entertainment",
        "music": "entertainment",
        "podcast": "entertainment",
        "media": "entertainment",
        "literature": "entertainment",
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
    target_class = categorize_text(query, list(TARGET_MAP.keys()))[0]
    return TARGET_MAP[target_class[0]]


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
