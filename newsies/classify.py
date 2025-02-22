import torch
from transformers import pipeline
from newsies.chroma_client import TAGSDB as tags_db

device_str = (
    # f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"
    f"cuda"
    if torch.cuda.is_available()
    else "cpu"
)
categorizer = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli", device=device_str
)


THEME_MAP = {}
_default_themes = {
    "detailed news story": "DOCUMENT",
    "read an article": "DOCUMENT",
    "any news stories": "DOCUMENT",
    "list headlines about a common person, place, or thing": "DOCUMENT",
    "people, places, things in the news": "ENTITY",
    "summary of multiple new stories": "SUMMARY",
    "common themes in multiple stories": "SUMMARY",
}


def embed_themes(theme_map: dict):
    """
    embed_themes
     - upsert themes as embeddings to ChromaDB
    """
    tags_db.embed_documents(
        document_ids=[k.replace(" ", "_") for k in theme_map.keys()],
        docs=[k for k in theme_map.keys()],
        metadata=[{"theme": v} for v in theme_map.values()],
    )


# Ensure the tags database has at least the default tags
embed_themes(_default_themes)


def refresh_themes():
    """
    refresh_themes
     - load all of the themes from the ChromaDB collection

     These can be updated over time to identify preferred classifications.  As these
     are used in prompts analysis, they can customize retrieval based on user idioms
    """
    _all_themes = tags_db.collection.get()
    _theme_count = len(_all_themes["documents"])
    global THEME_MAP

    THEME_MAP = {
        _all_themes["documents"][i]: _all_themes["metadatas"][i]["theme"]
        for i in range(_theme_count)
    }


refresh_themes()


# Function to classify text using zero-shot classification
def categorize_text(text, labels):

    result = categorizer(text, labels)
    categories = result["labels"]
    return categories


def prompt_analysis(query) -> str:
    """
    prompt_analysis:
      - Analyze a prompt
    """

    theme = THEME_MAP[categorize_text(query, list(THEME_MAP.keys()))[0]]

    categories = categorize_text(
        query,
        [
            "world-news",
            "us-news",
            "politics",
            "business",
            "technology",
            "science",
            "health",
            "entertainment",
            "sports",
            "oddities",
        ],
    )[:3]
    return {"theme": theme, "categories": categories}
