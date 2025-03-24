"""
newsies.ap_news.summary_visistory
"""

from typing import List

import torch
import torch.multiprocessing as mp
from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
)

from .article import Article


# pylint: disable=broad-exception-caught, too-many-locals, global-statement


mp.set_start_method("spawn", force=True)

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Pegasus model and tokenizer
# You can choose a different model based on your dataset (e.g., 'google/pegasus-large')
MODEL_NAME = "google/pegasus-large"
MODEL = None
TOKENIZER = None


class SummaryVisitor:
    """
    SummaryVisitor
    """

    def visit(self, node: any):
        """
        visit
        """
        node.accept(self)

    def visit_article(self, article: Article):
        """
        visit_article
        """
        article.summary = summarize_story(article.story)

    def summarize(self, text: str) -> str:
        """
        summarize
        """

        summary: str = text[:100]
        # Implement a summarization algorithm
        return summary


def summarize_chunk(chunk, max_length=200):
    """
    summarize_chunk

    """

    # Lazy-load the model at run-time
    global MODEL, TOKENIZER
    if MODEL is None:
        MODEL = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(
            DEVICE_STR
        )
    if TOKENIZER is None:
        TOKENIZER = PegasusTokenizer.from_pretrained(MODEL_NAME)
    inputs = TOKENIZER(chunk, return_tensors="pt", max_length=1024, truncation=True).to(
        DEVICE_STR
    )
    summary_ids = MODEL.generate(
        **inputs, max_length=max_length
    )  # Output summary length limit
    return TOKENIZER.decode(summary_ids[0], skip_special_tokens=True)


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


def summarize_story(text: str, max_tokens: int = 800):
    """
    Summarizes a news article while handling token limits using a sliding window
    for long paragraphs.

    Parameters:
        text (str): The article's text.
        max_tokens (int, optional): Maximum token limit per chunk (default: 800).

    Returns:
        str: The summarized story.
    """
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = PegasusTokenizer.from_pretrained(MODEL_NAME)

    # Step 1: Split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    summarized_paragraphs = []

    # Step 2: Process each paragraph
    for paragraph in paragraphs:
        token_count = len(
            TOKENIZER.encode(paragraph, add_special_tokens=False)
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
