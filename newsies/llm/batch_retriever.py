"""
newsies.llm.batch_retriever
This module contains the BatchRetriever class, which is used to retrieve batches of
articles from ChromaDB.  This is a Visistor class targetting the BatchSet class.
"""

import json
from typing import Any

from newsies.chromadb_client import ChromaDBClient
from newsies.collections import NEWS
from newsies.llm.batch_set import BatchSet


class BatchRetriever:
    """
    BatchRetriever class is used to retrieve batches of articles from ChromaDB.
    This is a Visitor class targeting the BatchSet class.
    """

    def visit(self, batch_set: Any):
        """
        Visit the BatchSet class and retrieve batches of articles from ChromaDB.
        """
        match batch_set:
            case BatchSet():
                self.visit_batch_set(batch_set)
            case _:
                raise TypeError(
                    f"BatchRetriever only accepts BatchSet, got {type(batch_set)}"
                )

    def visit_batch_set(self, batch_set):
        """
        Visit the BatchSet class and retrieve batches of articles from ChromaDB.
        """
        client = ChromaDBClient()
        client.collection_name = NEWS
        for batch in batch_set:
            results = client.collection.get(
                ids=list(batch), include=["metadatas", "embeddings"]
            )
            batch_set.embeddings.append(results["embeddings"])
            batch_set.metadatas.append(results["metadatas"])
