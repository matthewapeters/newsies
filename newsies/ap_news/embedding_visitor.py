"""
newsies.ap_news.embedding_visitor
"""

from datetime import datetime
import json
import math
from typing import Dict, List

import torch
from sentence_transformers import SentenceTransformer

from newsies.ap_news.article import Article
from newsies.collections import TAGS
from newsies.chromadb_client import ChromaDBClient

from .sections import SECTIONS

# pylint: disable=broad-exception-caught

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2", device=DEVICE_STR
)  # Fast and good quality

news_sections = {s for s in SECTIONS if s != ""}


class EmbeddingVisitor:
    """
    EmbeddingVisitor
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
        if __name__ in article.pipelines:
            return
        to_embed = list(article.section_headlines.values())
        to_embed.append(article.story)
        to_embed.append(article.summary)
        to_embed.extend(article.named_entities)
        embeddings: List[float] = []
        for te in to_embed:
            embeddings.append(embedding_model.encode(te))
        article.embeddings = embeddings
        article.pipelines[__name__] = datetime.now().isoformat()

    def compute_tfidf(self):
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
        section_doc_counts = self.get_section_doc_counts()
        all_ngrams = client.collection.get()

        total_docs = sum(
            section_doc_counts.values()
        )  # Total documents across all sections
        for ngram_id, metadata in zip(all_ngrams["ids"], all_ngrams["metadatas"]):
            metadata["tfidf"] = {}

            sections = json.loads(metadata["sections"])
            num_sections = len(
                sections
            )  # In how many sections does this n-gram appear?
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
                    idf = math.log(
                        total_docs / num_sections
                    )  # Inverse Document Frequency
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

    def get_section_doc_counts(self) -> Dict[str, int]:
        """
        Returns a dictionary of {section: document_count} by querying ChromaDB.

        Returns:
            dict: {section: document count}
        """
        section_counts = {}
        client = ChromaDBClient()
        full_count = client.collection.count()

        for section in news_sections:
            # Query ChromaDB for documents matching this section
            result = client.collection.get(
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
