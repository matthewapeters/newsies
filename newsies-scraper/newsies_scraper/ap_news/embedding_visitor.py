"""
newsies.ap_news.embedding_visitor

    Embeds the contents of the article for similarity comparison

"""

from datetime import datetime
import json
import math
from typing import Any, Dict, List, Union

import torch
from torch import Tensor
from sentence_transformers import SentenceTransformer
import numpy as np

from ..ap_news.article import Article
from newsies_clients.collections import TAGS
from newsies_clients.chromadb_client import ChromaDBClient

from .sections import SECTIONS

# pylint: disable=broad-exception-caught, global-statement

embedding_model: SentenceTransformer = None

news_sections = {s for s in SECTIONS if s != ""}


def _init_():
    global embedding_model
    models = ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if embedding_model is None:
        embedding_model = SentenceTransformer(
            models[0], device=device_str
        )  # Fast and good quality - reverse embedding look up?


class EmbeddingVisitor:
    """
    EmbeddingVisitor
    """

    def visit(self, node: Union[Article, Any]):
        """
        visit
        """
        node.accept(self)

    def visit_article(self, article: Article):
        """
        visit_article
        """
        _init_()
        if __name__ in article.pipelines:
            return
        to_embed = [article.story]
        to_embed.extend([f"{k}: {v}" for k, v in article.section_headlines.items()])
        to_embed.extend(article.named_entities)
        doc = " ".join(to_embed)
        tensor_embedding = embedding_model.encode(doc)
        match tensor_embedding:
            case Tensor():
                if tensor_embedding.dim() == 2 and tensor_embedding.shape[0] == 1:
                    tensor_embedding = tensor_embedding.squeeze(
                        0
                    )  # remove batch dimension
                tensor_embedding = tensor_embedding.detach().cpu().tolist()
            case np.ndarray():
                if tensor_embedding.shape[0] == 1:
                    tensor_embedding = tensor_embedding.squeeze(0)
                tensor_embedding = tensor_embedding.tolist()

        article.embeddings = tensor_embedding
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
