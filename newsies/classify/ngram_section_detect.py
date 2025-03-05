"""
newsies.classify.ngram_section_detect
"""

from typing import List
import json
from collections import defaultdict

from sentence_transformers import SentenceTransformer

from newsies.collections import TAGS
from newsies.chromadb_client import ChromaDBClient


def get_relevant_sections(
    prompt: str, top_k: int = 3, return_top: int = 2000
) -> List[str]:
    """
    Queries ChromaDB to find the most relevant sections for a given prompt.
    - Uses embeddings to retrieve similar ngrams.
    - Aggregates TF-IDF scores for each section.
    - Returns the top `top_k` sections with highest cumulative TF-IDF scores.

    :param prompt: User input query.
    :param chroma_client: Instance of the ChromaDB client.
    :param top_k: Number of top relevant sections to return.
    :param return_top: number of closest ngrams to analyze
    :return: List of most relevant section names.
    """

    # Load embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert prompt to embedding
    prompt_embedding = embedding_model.encode(prompt).tolist()
    chroma_client = ChromaDBClient()
    chroma_client.collection_name = TAGS
    # Query ChromaDB for similar ngrams
    results = chroma_client.collection.query(
        query_embeddings=[prompt_embedding],
        n_results=return_top,  # Fetch top 10 closest ngrams
    )

    # Aggregate TF-IDF scores by section
    section_scores = defaultdict(float)

    for metadata in results["metadatas"][0]:  # Assuming single query, so index [0]
        if "tfidf" in metadata:
            section_tfidf = json.loads(metadata["tfidf"])  # Convert back from JSON
            for section, tfidf in section_tfidf.items():
                if section not in [
                    "",
                    "oddities",
                ]:  # front page and oddities are ambiguous catch-alls
                    section_scores[
                        section
                    ] += tfidf  # Sum TF-IDF scores for each section

    # Sort sections by total TF-IDF score (descending)
    sorted_sections = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)

    # Return top-k sections
    return [section for section, _ in sorted_sections[:top_k]]
