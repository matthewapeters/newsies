"""
newsies.classification_heuristics
"""

from newsies.targets import DOCUMENT, HEADLINE, PAGE, SUMMARY
from newsies.actions import READ, LIST, COUNT  # , SYNTHESIZE


NEWS_SECTION_HEURISTICS = {
    "front page": "",
    "top story": "",
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
