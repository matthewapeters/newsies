from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

from newsies.ap_news.main import get_latest_news, get_article
from newsies.chroma_client import CRMADB
from newsies.chromadb_client.main import ChromaDBClient
from newsies.classify import prompt_analysis
from newsies.llm import LLM as llm, identify_themes, identify_entities
from newsies.summarizer import summarize_story


def news_summarizer(headlines: dict):
    """
    news_summarizer:
      - Summarize news articles
    """

    def process_story(k, v):
        doc_id = k + "_summary"
        metadata = v.copy()
        metadata["text"] = summarize_story(v["uri"], CRMADB, doc_id)
        metadata["type"] = "summary"
        print(f"Summarized: {k}")
        CRMADB.add_documents({doc_id: metadata})

    with ThreadPoolExecutor(
        max_workers=4
    ) as executor:  # Adjust max_workers based on CPU
        executor.map(lambda kv: process_story(*kv), headlines.items())


def news_loader(headlines: dict) -> dict:
    """
    news_loader:
      - Load news articles
    """
    with Pool(processes=8) as ppool:
        ppool.map(
            get_article,
            [(v["url"], v["headlines"], v["categories"]) for v in headlines.values()],
        )

    CRMADB.add_documents(headlines)
