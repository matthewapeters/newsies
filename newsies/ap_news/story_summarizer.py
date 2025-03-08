"""
newsies.summarizer
"""

import pickle
from datetime import datetime
import re
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.multiprocessing as mp
from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
)

from newsies.chroma_client import CRMADB
from newsies.chromadb_client import ChromaDBClient
from newsies.targets import SUMMARY
from newsies.ap_news.latest_news import path

from ..document_structures import Document


# pylint: disable=broad-exception-caught, too-many-locals


mp.set_start_method("spawn", force=True)

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Pegasus model and tokenizer
# You can choose a different model based on your dataset (e.g., 'google/pegasus-large')
MODEL_NAME = "google/pegasus-large"
model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE_STR)
tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)


def summarize_chunk(chunk, max_length=200):
    """
    summarize_chunk

    """
    inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True).to(
        DEVICE_STR
    )
    summary_ids = model.generate(
        **inputs, max_length=max_length
    )  # Output summary length limit
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def read_story(uri) -> str:
    """read_story"""
    text = "DOCUMENT NOT AVAILABLE"
    try:
        with open(uri, "r", encoding="utf8") as f:
            text = f.read()
        # remove AP credit tags for Video or photos (and related caption) , as they are not in the text
        # remove the AP legal at the end so it does not confuse the summary
        text = re.sub(
            r".*\(AP \w*\/\w* \w*\)|The Associated Press.*$|.*AP is solely responsible.*$|\n{2,}",
            "\n",
            text,
        )
        text = text.replace("\n\n", "\n")
        # remove quotes and capitalization so we get a more balanced summary of the story
        # quotes tend to super-cede other ideas in the story
        text = (
            text.replace('"', "")
            .replace("”", "")
            .replace("“", "")
            .replace("' ", " ")
            .lower()
        )
    except Exception:
        pass
    return text


def tokenize_text(text: str) -> int:
    """Returns the token count for a given text."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_text_with_overlap(
    text: str, max_tokens: int, overlap_ratio: float = 0.75
) -> List[str]:
    """
    Splits text into chunks using a sliding window with overlap if it exceeds max_tokens.

    Parameters:
        text (str): The input text to split.
        max_tokens (int): The maximum allowed token count per chunk.
        overlap_ratio (float): The percentage of overlap between consecutive chunks.

    Returns:
        List[str]: List of text chunks.
    """
    words = text.split()  # Simple word-based split
    chunks = []
    start = 0
    step = int(max_tokens * (1 - overlap_ratio))  # Compute step size based on overlap

    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):  # Stop if we reach the end
            break

        start += step  # Move forward with overlap

    return chunks


def summarize_story(uri: str, chroma_client, doc_id: str = None, max_tokens: int = 800):
    """
    Summarizes a news article while handling token limits using a sliding window
    for long paragraphs.

    Parameters:
        uri (str): The article's URI.
        chroma_client: ChromaDB client instance for caching.
        doc_id (str, optional): Document ID for caching results.
        max_tokens (int, optional): Maximum token limit per chunk (default: 800).

    Returns:
        str: The summarized story.
    """
    if doc_id:
        try:
            cached_summary = chroma_client.collection.get(
                include=["documents"], ids=[doc_id]
            )
            if cached_summary["documents"]:
                return cached_summary["documents"][0]  # Return cached result
        except Exception:
            pass  # Ignore errors if no result

    text = read_story(uri)  # Assume this fetches raw article text

    # Step 1: Split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    summarized_paragraphs = []

    # Step 2: Process each paragraph
    for paragraph in paragraphs:
        token_count = len(
            tokenizer.encode(paragraph, add_special_tokens=False)
        )  # tokenize_text(paragraph)

        if token_count <= max_tokens:
            # If paragraph fits within token limit, summarize it directly
            summarized_paragraphs.append(summarize_chunk(paragraph, max_tokens))
        else:
            # Step 3: If too long, use a sliding window
            chunks = split_text_with_overlap(paragraph, max_tokens)
            chunk_summaries = [summarize_chunk(chunk, max_tokens) for chunk in chunks]
            merged_paragraph = " ".join(chunk_summaries)

            summarized_paragraphs.append(
                summarize_chunk(merged_paragraph)
            )  # Re-summarize

    # Step 4: Merge and summarize the full document
    merged_summary = " ".join(summarized_paragraphs)
    document_chunks = split_text_with_overlap(merged_summary, max_tokens)
    final_summaries = [summarize_chunk(chunk) for chunk in document_chunks]

    return " ".join(final_summaries)


def process_batch_news_summarizer_story(k: str, v: Document, archive: str = None):
    """
    process_story
    """
    doc_id = k + "_summary"
    metadata = Document(**v.dump())
    metadata.text = summarize_story(v.uri, CRMADB, doc_id)
    metadata.target = SUMMARY
    print(f"Summarized: {k}")
    client = ChromaDBClient()
    if archive is None:
        client.collection_name = CRMADB.collection_name
    else:
        client.collection_name = f"ap_news_{archive}"
    client.add_documents({doc_id: metadata})


def batch_news_summarizer(documents: Dict[str, Document] = None, archive: str = None):
    """
    news_summarizer
    Summarizes multiple news articles concurrently using ProcessPoolExecutor.
    """
    if documents is None:
        pikl_path = path("latest_news", archive).replace(".txt", ".pkl")
        with open(pikl_path, "rb") as fh:
            documents = pickle.load(fh)

    if archive is None:
        archive = datetime.now().strftime(r"%Y-%m-%d")

    max_workers = 2
    print(f"\t   - {max_workers} workers\n")
    with ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:  # Use multiprocessing for efficiency
        futures = [
            executor.submit(process_batch_news_summarizer_story, k, v, archive)
            for k, v in documents.items()
        ]
        for future in futures:
            future.result()  # Ensures all tasks complete before exiting
