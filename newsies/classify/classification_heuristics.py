"""
newsies.classification_heuristics
"""

from newsies.actions import READ, LIST, COUNT  # , SYNTHESIZE
from newsies.targets import DOCUMENT, HEADLINE, PAGE, SUMMARY
from newsies.chromadb_client import ChromaDBClient

# pylint: disable=global-statement, unused-argument, fixme

TAGS_COLLECTION = "newsies_tags"

NEWS_SECTION_HEURISTICS = {
    "front page": "",
    "top story": "",
    "africa": "world-news",
    "asia": "world-news",
    "australia": "world-news",
    "canada": "world-news",
    "china": "world-news",
    "europe": "world-news",
    "france": "world-news",
    "germany": "world-news",
    "global news": "world-news",
    "international news": "world-news",
    "ireland": "world-news",
    "mexico": "world-news",
    "norway": "world-news",
    "south america": "world-news",
    "uk": "world-news",
    "united kingdom": "world-news",
    "world news": "world-news",
    "USA": "us-news",
    "domestic": "us-news",
    "in america": "us-news",
    "in the us": "us-news",
    "state": "us-news",
    "us news": "us-news",
    "congress": "politics",
    "president": "politics",
    "senate": "politics",
    "supreme court": "politics",
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

TARGET_HEURISTICS = {
    "story": DOCUMENT,
    "stories": DOCUMENT,
    "article": DOCUMENT,
    "articles": DOCUMENT,
    "details news story": DOCUMENT,
    "read story": DOCUMENT,
    "read article": DOCUMENT,
    "any news stories": DOCUMENT,
    "headline": HEADLINE,
    "title": HEADLINE,
    "headlines": HEADLINE,
    "titles": HEADLINE,
    "list headlines": HEADLINE,
    "list titles": HEADLINE,
    "read summary of article": SUMMARY,
    "read summary of story": SUMMARY,
    "the summary of article": SUMMARY,
    "the summary of story": SUMMARY,
    "page": PAGE,
    "next page": PAGE,
    "previous page": PAGE,
}


QUERY_OR_REFERENCE_HEURISTICS = {
    "refers to 'today'": "NEW",
    "refers to 'on this day'": "NEW",
    "refers to new query": "NEW",
    "introduces a change of topic": "NEW",
    "refers to prior query or prompt": "OLD",
    "refers somthing we talked about before": "OLD",
}

ACTION_HEURISTICS = {
    "read an article": READ,
    "pull an article": READ,
    "read a story": READ,
    "pull a story": READ,
    "list titles": LIST,
    "list headlines": LIST,
    "list stories": LIST,
    "list articles": LIST,
    "count articles": COUNT,
    "read the summary of an article": READ,
    "read the summary of a story": READ,
}

ONE_MANY_ALL_HEURISTICS = {
    "referring to one story or article, story, person, place or thing": "ONE",
    (
        "referring to more than one, but less than all articles, "
        "stories, people, places or things"
    ): "MANY",
    "referring to all articles, stories, people, places or things": "ALL",
    (
        "request is for the full set of articles, stories, people, "
        "places or things from a category"
    ): "ALL",
    "request if for each document from one or more categories": "ALL",
}


def embed_targets(target_map: dict):
    """
    embed_targets
     - upsert targets as embeddings to ChromaDB
    """
    # global TARGET_HEURISTICS
    # TARGET_HEURISTICS = target_map
    #
    # tags_db = ChromaDBClient()
    # TAGS_COLLECTION = "newsies_tags"
    # tags_db.collection_name = TAGS_COLLECTION
    # tags_db.language = "en"
    # tags_db.embed_documents(
    #    document_ids=[k.replace(" ", "_") for k in target_map.keys()],
    #    docs=[k for k in target_map.keys()],
    #    metadata=[{"target": v} for v in target_map.values()],
    # )


def refresh_targets():
    """
    refresh_targets
     - load all of the targets from the ChromaDB collection

     These can be updated over time to identify preferred classifications.  As these
     are used in prompts analysis, they can customize retrieval based on user idioms
    """

    #
    tags_db = ChromaDBClient()
    tags_db.collection_name = TAGS_COLLECTION
    tags_db.language = "en"
    all_targets = tags_db.collection.get()
    target_count = len(all_targets["documents"])
    global TARGET_HEURISTICS

    TARGET_HEURISTICS = {
        all_targets["documents"][i]: all_targets["metadatas"][i]["target"]
        for i in range(target_count)
    }


# TODO - uncomment when we want to retrieve the mappings from the chromadb
# Ensure the tags database has at least the default tags
# embed_targets(TARGET_HEURISTICS)
# refresh_targets()
