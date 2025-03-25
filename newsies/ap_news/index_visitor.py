"""
newsies.ap_news.index_visitor
"""

import os
from typing import Any, Union
from datetime import datetime

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from newsies.targets import DOCUMENT

from .article import Article

# pylint: disable=broad-exception-raised, unused-argument


CHROMA_HOST = "CHROMADB_HOST"
CHROMA_PORT = "CHROMADB_PORT"
CHROMA_CREDS = "CHROMADB_CREDS"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class IndexVisitor:
    """
    ChromaDBClient is a wrapper around the ChromaDB client that provides
    a convenient way to interact with the ChromaDB client.
    """

    # Load a Hugging Face embedding model
    _embed_model = SentenceTransformer(MODEL_NAME)

    def __init__(self, *args, **kwargs):
        self._client: chromadb.HttpClient = None
        self._collection: chromadb.Collection = None
        self._host: str = None
        self._port: int = None
        self._chroma_creds: str = None
        self._colection_name: str = None
        self._lang: str = None

        if "host" in kwargs:
            self._host = kwargs["host"]
        else:
            self._host = os.getenv(CHROMA_HOST, "localhost")
        if "port" in kwargs:
            self._port = kwargs["port"]
        else:
            self._port = os.getenv(CHROMA_PORT, "8000")
        if "chroma_creds" in kwargs:
            self._chroma_creds = kwargs["chroma_creds"]
        else:
            self._chroma_creds = os.getenv(CHROMA_CREDS)
        if self._chroma_creds == "":
            raise Exception("CHROMA_CREDS environment variable is not set")

        # Connect to ChromaDB
        self._client: chromadb.HttpClient = chromadb.HttpClient(
            host=self._host,
            port=self._port,
            settings=Settings(
                chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
                chroma_client_auth_credentials=self._chroma_creds,
            ),
        )

    @property
    def client(self) -> chromadb.HttpClient:
        """
        client
        """
        return self._client

    @property
    def language(self) -> str:
        """
        language
        """
        return self._lang

    @language.setter
    def language(self, lang: str):
        self._lang = lang

    @property
    def collection_name(self) -> str:
        """
        collection
        """
        return self._collection.name

    @collection_name.setter
    def collection_name(self, collection_name: str):
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(self._collection_name)
        if self._collection is None:
            raise Exception("FAILED TO CONNECT TO CHROMA DB")

    @property
    def collection(self) -> chromadb.Collection:
        """collection"""
        return self._collection

    def visit(self, node: Union[Article, Any]):
        """visit"""
        node.accept(self)

    def visit_article(self, article: Article):
        """visit_article"""
        if __name__ in article.pipelines:
            return

        document_ids = [article.item_id]
        metadatas = {
            "publish_date": article.publish_date.isoformat(),
            "authors": ", ".join(article.authors),
            "ner_terms": ", ".join(article.named_entities),
            "url": article.url,
            "target": DOCUMENT,
            **{
                f"section{i}": section
                for i, section in enumerate(article.section_headlines.keys())
            },
            **{
                f"headline{i}": headline
                for i, headline in enumerate(article.section_headlines.values())
            },
        }
        embeddings = article.embeddings
        params = {"ids": document_ids, "embeddings": embeddings, "metadatas": metadatas}

        self.collection.upsert(**params)
        article.pipelines[__name__] = datetime.now().isoformat()
