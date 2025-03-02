"""
newsies.batch_summarizer
"""

import re
from typing import List

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

from newsies.chromadb_client import ChromaDBClient


DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Pegasus model and tokenizer
MODEL_NAME = "google/pegasus-large"
model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE_STR)
tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)


def split_text(text, max_tokens=800, overlap=200):
    """
    split_text
    Splits text into overlapping chunks."""
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(tokens[start:end])
        start += max_tokens - overlap  # Overlap for context

    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]


def batch_summarize(chunks: List[str], max_length=200, batch_size=4):
    """
    batch_summarize
    Summarizes multiple text chunks at once for efficiency.
    """
    summaries = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True
        ).to(DEVICE_STR)
        with torch.no_grad():
            summary_ids = model.generate(**inputs, max_length=max_length)
        summaries.extend(tokenizer.batch_decode(summary_ids, skip_special_tokens=True))
    return summaries


def read_story(uri) -> str:
    """Reads and preprocesses a story from a file."""
    with open(uri, "r", encoding="utf8") as f:
        text = f.read()

    # Remove unwanted tags, AP legal text, and excessive line breaks
    text = re.sub(
        r".*\(AP \w*\/\w* \w*\)|The Associated Press.*$|.*AP is solely responsible.*$|\n{2,}",
        "\n",
        text,
    ).replace("\n\n", "\n")

    # Normalize quotes and capitalization for balanced summaries
    text = (
        text.replace('"', "")
        .replace("”", "")
        .replace("“", "")
        .replace("' ", " ")
        .lower()
    )
    return text


def summarize_story(uri: str, chroma_client: ChromaDBClient, doc_id: str = None):
    """
    summarize_story
    Summarizes a news article, using cached summaries if available.
    """
    if doc_id:
        cached_summary = chroma_client.collection.get(
            include=["documents"], ids=[doc_id]
        )
        if len(cached_summary["documents"]) > 0:
            return cached_summary["documents"][0]  # Return cached result

    text = read_story(uri)

    # Split text into chunks for batch processing
    document_chunks = [c for p in text.split("\n") if len(p) > 1 for c in split_text(p)]

    # Batch summarize chunks
    summaries = batch_summarize(document_chunks, max_length=200, batch_size=4)
    merged_summary = " ".join(summaries)

    # Further summarize if needed
    document_chunks = split_text(merged_summary, max_tokens=800)
    summary = " ".join(batch_summarize(document_chunks, max_length=200, batch_size=4))

    return summary
