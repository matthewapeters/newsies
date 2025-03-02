"""
newsies.classify.train
"""

import re
from collections import Counter
from transformers import AutoTokenizer
from nltk.util import ngrams
from nltk.corpus import stopwords
import nltk

from newsies.chromadb_client import ChromaDBClient
from .classification_heuristics import TAGS_COLLECTION

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load tokenizer (change model as needed)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def extract_ngrams(text, n=30):
    """Extract n-grams up to length n from text, filtering stopwords and punctuation."""
    words = [
        w.lower() for w in re.findall(r"\b\w+\b", text) if w.lower() not in stop_words
    ]
    ngram_freq = Counter()

    for size in range(1, n + 1):
        for ng in ngrams(words, size):
            ngram_freq[" ".join(ng)] += 1

    return ngram_freq


def store_keywords_in_chromadb(story, section, client: ChromaDBClient, n=30):
    """Extract n-grams specific to a section and store them in ChromaDB."""
    ngram_counts = extract_ngrams(story, n)

    client.collection_name = TAGS_COLLECTION

    for ngram, freq in ngram_counts.items():
        tokenized_ngram = tokenizer.encode(ngram, add_special_tokens=False)
        client.collection.add(
            ids=[f"{section}_{ngram}"],
            metadatas=[{"section": section, "ngram": ngram, "frequency": freq}],
            embeddings=[tokenized_ngram],
        )

    print(f"Stored {len(ngram_counts)} n-grams for section: {section}")
