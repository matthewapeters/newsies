"""
newsies.summarizer
"""

import re

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

from newsies.chromadb_client import ChromaDBClient

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"


# Load the Pegasus model and tokenizer
# You can choose a different model based on your dataset (e.g., 'google/pegasus-large')
MODEL_NAME = "google/pegasus-large"
model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE_STR)
tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)


def split_text(text, max_tokens=800, overlap=200):
    """split_text"""
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(tokens[start:end])
        start += max_tokens - overlap  # Overlap for context

    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]


def summarize_chunk(chunk, max_length=200):
    """summarize_chunck"""
    inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True).to(
        DEVICE_STR
    )
    summary_ids = model.generate(
        **inputs, max_length=max_length
    )  # Output summary length limit
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def read_story(uri) -> str:
    """read_story"""
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
    return text


def summarize_story(uri: str, chroma_client: ChromaDBClient, doc_id: str = None):
    """
    summarize_story:
      - Summarize a news article
    """
    if doc_id:
        cached_summary = chroma_client.collection.get(
            include=["documents"], ids=[doc_id]
        )
        if cached_summary:
            return cached_summary["documents"][0]  # Return cached result

    text = read_story(uri)

    # summarize each paragraph
    document_chunks = [c for p in text.split("\n") if len(p) > 1 for c in split_text(p)]
    summaries = [summarize_chunk(chunk, len(chunk) / 2) for chunk in document_chunks]
    merged_summary = " ".join(summaries)

    # summarize the summaries (by chunks)
    document_chunks = split_text(merged_summary, max_tokens=800)
    summaries = [summarize_chunk(chunk) for chunk in document_chunks]
    summary = " ".join(summaries)

    return summary
