"""
newsies.classify.train
"""

import json
import math
import re
from typing import List
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import stopwords
import nltk
from sentence_transformers import SentenceTransformer

from newsies.ap_news import SECTIONS
from newsies.chromadb_client import ChromaDBClient
from newsies.chroma_client import CRMADB
from newsies.document_structures import Document
from newsies.collections import TAGS


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast and good quality


news_sections = {s for s in SECTIONS if s != ""}

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def extract_ngrams(text: str, n=30):
    """Extract n-grams up to length n from text, filtering stopwords and punctuation."""
    words = [
        w.lower() for w in re.findall(r"\b\w+\b", text) if w.lower() not in stop_words
    ]
    ngram_freq = Counter()

    for size in range(1, n + 1):
        for ng in ngrams(words, size):
            ngram_freq[" ".join(ng)] += 1

    return ngram_freq


def analyze_ngrams_per_section(headlines):
    """
    analyze_ngrams_per_section
      - read each of the news stories
    """
    v: Document = None
    for v in headlines.values():
        with open(v.uri, "r", encoding="utf8") as fh:
            text = fh.read()
        store_keywords_in_chromadb(story=text, sections=v.sections)


def store_keywords_in_chromadb(story: str, sections: List[str], n=30, min_freq=2):
    """Extracts and updates section-specific n-gram frequencies in ChromaDB."""
    chroma_client: ChromaDBClient = ChromaDBClient()
    chroma_client.collection_name = TAGS

    ngram_counts = extract_ngrams(story, n)
    common_ngrams = ngram_counts.most_common()
    filtered_ngrams = [
        (ngram, freq) for ngram, freq in common_ngrams if freq >= min_freq
    ]

    for ngram, freq in filtered_ngrams:
        ngram_id = f"ngram_{ngram}"
        existing_data = chroma_client.collection.get(ids=[ngram_id])

        if existing_data["documents"]:  # N-gram exists
            metadata = existing_data["metadatas"][0]
            section_freqs_raw = metadata.get("sections", "{}")
            section_freqs = json.loads(section_freqs_raw)

            # Update section frequency
            for section in sections:
                section_freqs[section] = section_freqs.get(section, 0) + freq

            # Remove sections where frequency is too low
            section_freqs = {
                sec: f for sec, f in section_freqs.items() if f >= min_freq
            }

            if not section_freqs:  # Remove n-gram entirely if it has no strong section
                chroma_client.collection.delete(ids=[ngram_id])
                continue

            metadata["sections"] = json.dumps(section_freqs)
            metadata["most_likely_section"] = max(section_freqs, key=section_freqs.get)

            chroma_client.collection.update(ids=[ngram_id], metadatas=[metadata])

        else:  # New n-gram
            section_metadata = json.dumps({section: freq for section in sections})
            metadata = {
                "ngram": ngram,
                "sections": section_metadata,
                "most_likely_section": sections[0],
            }
            # Generate proper embeddings
            embedding = embedding_model.encode(ngram).tolist()

            chroma_client.collection.add(
                ids=[ngram_id],
                metadatas=[metadata],
                embeddings=[embedding],
            )

    print(
        f"Stored/Updated {len(filtered_ngrams)} n-grams for sections: {", ".join(sections)}"
    )


def compute_tfidf():
    """
    Computes TF-IDF scores for stored n-grams per section.

    Args:
        client: ChromaDB client instance.
    """
    client = ChromaDBClient()
    client.collection_name = TAGS

    # collection = client.get_collection("news_keywords")
    section_doc_counts = get_section_doc_counts()
    all_ngrams = client.collection.get()

    total_docs = sum(section_doc_counts.values())  # Total documents across all sections

    for ngram_id, metadata in zip(all_ngrams["ids"], all_ngrams["metadatas"]):
        metadata["tfidf"] = {}

        num_sections = len(
            metadata["sections"]
        )  # In how many sections does this n-gram appear?

        for section, freq in metadata["sections"].items():
            section_doc_count = section_doc_counts.get(section, 1)  # Avoid div-by-zero

            # Compute TF-IDF
            tf = freq / section_doc_count  # Term Frequency (normalized)
            idf = math.log(total_docs / num_sections)  # Inverse Document Frequency
            metadata["tfidf"][section] = tf * idf

        # Assign the best section based on highest TF-IDF
        metadata["most_likely_section"] = max(
            metadata["tfidf"], key=metadata["tfidf"].get
        )

        client.collection.update(ids=[ngram_id], metadatas=[metadata])

    print("Updated per-section TF-IDF scores using document counts.")


def get_section_doc_counts():
    """
    Returns a dictionary of {section: document_count} by querying ChromaDB.

    Returns:
        dict: {section: document count}
    """
    section_counts = {}
    full_count = CRMADB.collection.count()

    for section in news_sections:
        # Query ChromaDB for documents matching this section
        result = CRMADB.collection.get(
            where={
                "$or": [
                    {"section0": {"$eq": section}},
                    {"section1": {"$eq": section}},
                    {"section2": {"$eq": section}},
                ]
            },
            limit=full_count,
        )
        section_counts[section] = len(result["ids"])

    return section_counts
