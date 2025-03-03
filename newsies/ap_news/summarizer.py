"""
newsies.ap_news.summarizer
"""

from typing import Dict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

from newsies.chroma_client import CRMADB
from newsies.chromadb_client import ChromaDBClient
from newsies.summarizer import summarize_story
from newsies.targets import SUMMARY

from ..document_structures import Document


def news_summarizer(documents: Dict[str, Document]):
    """
    news_summarizer:
      - Summarize news articles
    """

    def process_story(k: str, v: Document):
        doc_id = k + "_summary"
        metadata = Document(**v.dump())
        # summarize_story will defer to existing summary in chromadb if it exists
        metadata.text = summarize_story(v.uri, CRMADB, doc_id)
        metadata.target = SUMMARY
        print(f"Summarized: {k}")
        client = ChromaDBClient()
        client.collection_name = CRMADB.collection_name
        client.add_documents({doc_id: metadata})

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_story, k, v) for k, v in documents.items()]
        for future in futures:
            future.result()  # Ensures all tasks complete before exiting


def process_batch_news_summarizer_story(k: str, v: Document):
    """
    process_story
    """
    doc_id = k + "_summary"
    metadata = Document(**v.dump())
    metadata.text = summarize_story(v.uri, CRMADB, doc_id)
    metadata.target = "SUMMARY"
    print(f"Summarized: {k}")
    client = ChromaDBClient()
    client.collection_name = CRMADB.collection_name
    client.add_documents({doc_id: metadata})


def batch_news_summarizer(documents: Dict[str, Document]):
    """
    news_summarizer
    Summarizes multiple news articles concurrently using ProcessPoolExecutor.
    """

    with ProcessPoolExecutor(
        max_workers=2
    ) as executor:  # Use multiprocessing for efficiency
        futures = [
            executor.submit(process_batch_news_summarizer_story, k, v)
            for k, v in documents.items()
        ]
        for future in futures:
            future.result()  # Ensures all tasks complete before exiting
