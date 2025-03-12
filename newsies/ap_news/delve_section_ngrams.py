"""
newsies.classify.train
"""

import pickle
import json
import math
import re
from typing import List, Dict

from collections import Counter
from nltk.util import ngrams
from nltk.corpus import stopwords

import spacy
from spacy.tokens import Doc

from sentence_transformers import SentenceTransformer
import torch

from newsies.ap_news import SECTIONS
from newsies.ap_news.latest_news import path
from newsies.chromadb_client import ChromaDBClient
from newsies.chroma_client import CRMADB
from newsies.document_structures import Document
from newsies.collections import TAGS
from newsies import targets

# pylint: disable=broad-exception-caught

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2", device=DEVICE_STR
)  # Fast and good quality


# Load spaCy model for English
nlp = spacy.load("en_core_web_sm")


news_sections = {s for s in SECTIONS if s != ""}

# nltk.download("stopwords")
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


def analyze_ngrams_per_section(headlines=None, archive: str = None):
    """
    analyze_ngrams_per_section
      - read each of the news stories
    """

    if headlines is None:
        pikl_path = path("latest_news", archive).replace(".txt", ".pkl")
        with open(pikl_path, "rb") as fh:
            headlines = pickle.load(fh)

    v: Document = None
    chroma_client: ChromaDBClient = ChromaDBClient()
    chroma_client.collection_name = TAGS
    texts: List[str] = []
    story_sections: List[List[str]] = []
    texts_len = 0
    file_count = 0
    buff_threshold = 40000
    for v in headlines.values():
        try:
            with open(v.uri, "r", encoding="utf8") as fh:
                text = fh.read()
                texts_len += len(text)
                texts.append(text)
            story_sections.append([s for s in v.sections if s != ""])
            file_count += 1
        except Exception as e:
            print(f"WARNING: {e}: {v.uri} not found - skipping")

        if texts_len >= buff_threshold:
            print(f"\t- processing {file_count} stories")
            store_keywords_in_chromadb(
                chroma_client=chroma_client, stories=texts, sections=story_sections
            )
            texts_len = 0
            texts.clear()
            story_sections.clear()
            file_count = 0
    if texts:
        print(f"\t- processing {file_count} stories")
        store_keywords_in_chromadb(
            chroma_client=chroma_client, stories=texts, sections=story_sections
        )


def generate_ngrams(
    doc: Doc, max_ngram_len: int = 5, min_freq: int = 1
) -> Dict[str, int]:
    """
    generate_ngrams
    """
    words: List[str] = []
    # Apply Named Entity Removal
    for token in doc:
        if token.ent_type_:  # If token is part of a named entity, use entity label
            words.append(f"[{token.ent_type_}]")

        elif (
            token.is_alpha and token.text.lower() not in stop_words
        ):  # remove stopwords
            words.append(token.text.lower())

    ngram_counts = {}

    # Generate n-grams
    for i in range(len(words)):
        for j in range(i + 1, min(i + max_ngram_len + 1, len(words) + 1)):
            ngram = " ".join(words[i:j])
            if ngram in ngram_counts:
                ngram_counts[ngram] += 1
            else:
                ngram_counts[ngram] = 1

    # Filter n-grams by min_freq
    return {ngram: count for ngram, count in ngram_counts.items() if count >= min_freq}


def store_keywords_in_chromadb(
    chroma_client: ChromaDBClient,
    stories: List[str],
    sections: List[List[str]],
    n: int = 30,
    min_freq: int = 2,
    batch_size: int = 1000,
):
    """
    Processes multiple texts and stores their meaningful n-grams in ChromaDB.

    - Supports batch processing.
    - Merges duplicate n-grams locally before database upsert.
    - Retrieves existing n-grams to merge frequencies properly.

    :param client: ChromaDB client connection.
    :param collection_name: Name of the ChromaDB collection.
    :param stories: List of texts to process.
    :param sections: List of section lists corresponding to each text.
    :param tokenizer: SentenceTransformer model for embeddings.
    :param n: Max n-gram length.
    :param min_freq: Minimum frequency for an n-gram to be stored.
    :param batch_size: Max batch size for database upserts.
    """
    batch = {}
    ne_weight = 2.0

    # Process each story
    for text, text_sections in zip(stories, sections):
        doc = nlp(text)

        named_ents = detect_named_entities(doc)
        assemble_batch_metadata(named_ents, text_sections, batch, ne_weight)

        filtered_ngrams = generate_ngrams(doc, n, min_freq)
        assemble_batch_metadata(filtered_ngrams, text_sections, batch, 1.0)

        # Upsert when batch size is reached
        if len(batch) >= batch_size:
            _upsert_batch(chroma_client, batch)
            batch.clear()  # Reset batch

    # Insert remaining items
    if batch:
        _upsert_batch(chroma_client, batch)


def assemble_batch_metadata(
    n_grams: iter, text_sections: List[str], batch: Dict[str, Dict], weight: float = 1.0
) -> Dict[str, Dict]:
    """
    assemble_batch_metadata
    """
    # Store n-grams with section frequencies
    for ngram, count in n_grams.items():
        ngram_id = f"ngram_{ngram}"

        if ngram in batch:
            # Update local section frequency counts
            for section in text_sections:
                batch[ngram]["metadata"]["sections"][section] = (
                    batch[ngram]["metadata"]["sections"].get(section, 0) + count
                )
        else:
            # Initialize new entry
            batch[ngram] = {
                "id": ngram_id,
                "metadata": {
                    "ngram": ngram,
                    "weight": weight,
                    "sections": {section: count for section in text_sections},
                    "most_likely_section": "",  # Placeholder; updated later
                },
                "embeddings": embedding_model.encode(ngram, add_special_tokens=False),
            }


def _upsert_batch(chroma_client: ChromaDBClient, batch: Dict[str, Dict]):
    """Handles merging of n-gram section frequencies before upserting to ChromaDB.

    - Retrieves existing n-gram metadata.
    - Merges section frequency counts locally.
    - Converts "sections" metadata field to JSON for ChromaDB storage.
    """

    existing_docs = chroma_client.collection.get(
        ids=[item["id"] for item in batch.values()]
    )

    # Extract existing section frequencies from stored metadata
    existing_data = {
        existing_docs["ids"][idx]: (
            json.loads(doc["sections"]) if "sections" in doc else {}
        )
        for idx, doc in enumerate(existing_docs["metadatas"])
    }

    for item in batch.values():
        ngram_id = item["id"]

        # Merge with existing section data if ngram exists in ChromaDB
        if ngram_id in existing_data:
            for section, count in existing_data[ngram_id].items():
                item["metadata"]["sections"][section] = (
                    item["metadata"]["sections"].get(section, 0) + count
                )

        # Convert sections dictionary to JSON string for ChromaDB storage
        item["metadata"]["sections"] = json.dumps(item["metadata"]["sections"])

    # Perform batch upsert
    chroma_client.collection.upsert(
        ids=[item["id"] for item in batch.values()],
        metadatas=[item["metadata"] for item in batch.values()],
        embeddings=[item["embeddings"] for item in batch.values()],
    )
    print(f"\t\t- batch upsert {len(batch)} ngrams")


def detect_named_entities(doc: Doc) -> Counter:
    """
    detect_named_entities
    """

    named_entities = Counter()
    for ent in doc.ents:
        if any(token.pos_ == "PROPN" for token in ent):
            named_entities[ent.text] += 1  # Count proper named entities
        else:
            # Remove stopwords from non-proper-noun entities
            tokens = [token.text for token in ent if not token.is_stop]
            if tokens:
                named_entities[" ".join(tokens)] += 1

    return named_entities


def compute_tfidf():
    """
    Computes TF-IDF scores for stored n-grams per section.

    Args:
        client: ChromaDB client instance.
    """
    client = ChromaDBClient()
    client.collection_name = TAGS
    batch_size = 1800
    ids: List[str] = []
    metas: List[Dict] = []

    # collection = client.get_collection("news_keywords")
    section_doc_counts = get_section_doc_counts()
    all_ngrams = client.collection.get()

    total_docs = sum(section_doc_counts.values())  # Total documents across all sections
    for ngram_id, metadata in zip(all_ngrams["ids"], all_ngrams["metadatas"]):
        metadata["tfidf"] = {}

        sections = json.loads(metadata["sections"])
        num_sections = len(sections)  # In how many sections does this n-gram appear?
        weight: float = metadata[
            "weight"
        ]  # Named Entities may be weighted higher than speech patterns

        if num_sections > 0:
            for section, freq in sections.items():
                section_doc_count = section_doc_counts.get(
                    section, 1
                )  # Avoid div-by-zero

                # Compute TF-IDF
                tf = freq / section_doc_count  # Term Frequency (normalized)
                idf = math.log(total_docs / num_sections)  # Inverse Document Frequency
                metadata["tfidf"][section] = tf * idf * weight

            # Assign the best section based on highest TF-IDF
            metadata["most_likely_section"] = max(
                metadata["tfidf"], key=metadata["tfidf"].get
            )
            metadata["tfidf"] = json.dumps(metadata["tfidf"])
            ids.append(ngram_id)
            metas.append(metadata)
            if len(ids) >= batch_size:
                client.collection.update(ids=ids, metadatas=metas)
                print(
                    f"\t- Updated {len(ids)} per-section TF-IDF scores using document counts."
                )
                ids.clear()
                metas.clear()
    if ids:
        client.collection.update(ids=ids, metadatas=metas)
        print(
            f"\t- Updated {len(ids)} per-section TF-IDF scores using document counts."
        )

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


def generate_named_entity_embeddings_for_stories(archive: str, target: str):
    """
    generate_named_entity_embeddings_for_stories
    """
    client = ChromaDBClient()
    client.collection_name = archive
    story_ids = client.collection.get(include=[], where={"target": {"$eq": target}})[
        "ids"
    ]
    story_count = len(story_ids)
    batch_size = 100
    for i in range(0, story_count, batch_size):
        ids = story_ids[i : min(i + batch_size, story_count)]
        batch = client.collection.get(ids=ids, include=["documents", "embeddings"])
        embed_map = {
            batch["ids"][idx]: (batch.get("embeddings", [None] * (idx + 1))[idx] or [])
            + [
                embedding_model.encode(ne)
                for ne in list(
                    detect_named_entities(nlp(batch["documents"][idx])).keys()
                )
            ]
            for idx in range(len(batch["ids"]))
        }
        client.collection.update(ids=embed_map.keys(), embeddings=embed_map.values())
