"""
newsies.chromadb.client
"""

import os
from typing import Dict, List, Union

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from newsies.document_structures import Document
from newsies.targets import DOCUMENT

# pylint: disable=broad-exception-raised, unused-argument


CHROMA_HOST = "CHROMADB_HOST"
CHROMA_PORT = "CHROMADB_PORT"
CHROMA_CREDS = "CHROMADB_CREDS"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class ChromaDBClient:
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
        self._client = chromadb.HttpClient(
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

    @property
    def collection(self) -> chromadb.Collection:
        """collection"""
        return self._collection

    def _upsert(
        self,
        document_ids: Union[List[str], str],
        docs: Union[List[str], str],
        embeddings: Union[List[float], float] = None,
        metadata: Union[List[str], str] = None,
        uris: Union[List[str], str] = None,
    ):
        """
        Add documents to the collection
        """
        max_docs = 20000
        len_docs = len(docs)
        for i in range(max_docs, len_docs + max_docs, max_docs):
            lower = i - max_docs
            upper = min(i, len_docs)

        params = {
            "ids": document_ids[lower:upper],
            "documents": docs[lower:upper],
            "embeddings": embeddings[lower:upper],
        }
        if uris is not None:
            params["uris"] = uris[lower:upper]
        if metadata is not None:
            params["metadatas"] = metadata[lower:upper]

            self.collection.upsert(**params)

    def embed_documents(
        self,
        document_ids: Union[List[str], str],
        docs: Union[List[str], str],
        metadata: Union[List[str], str] = None,
        uris: Union[List[str], str] = None,
    ):
        """
        Add documents to the collection
        """
        ci = "chunk_index"

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        for i, text in enumerate(docs):
            chunks = splitter.split_text(text)
            if len(chunks) > 1:
                print(f"Splitting document {i} into {len(chunks)} chunks")
                for j, chunk in enumerate(chunks):
                    if j == 0:
                        docs[i] = chunk
                        if metadata is not None:
                            metadata[i].update({ci: 0, "text": chunk})
                    else:
                        if uris is not None:
                            uris.append(uris[i] + f"_{j}")
                        docs.append(chunk)
                        document_ids.append(f"{document_ids[i]}_{j}")
                        if metadata is not None:
                            metadata.append({ci: j, "text": chunk, **metadata[i]})
            else:
                metadata[i].update({ci: 0})

        # Encode the documents and add the embedding model to the metadata
        embeddings = self._embed_model.encode(docs).tolist()

        params = {
            "document_ids": document_ids,
            "docs": docs,
            "embeddings": embeddings,
        }
        if uris is not None:
            params["uris"] = uris
        if metadata is not None:
            params["metadata"] = metadata

        self._upsert(**params)

    def add_documents(
        self,
        documents: Dict[str, Document],
    ):
        """
        Add documents to the collection

        @param documents: A dictionary of documents to add to the collection
            keyed by document ID
            Value is a dictionary with the following keys:
                - url: The URL of the document
                - uri: The URI of the document
                - date: The date of the document
                - source: The source of the document
                - headlines: A list of headlines for the document
                - sections: A list of sections from the news where the article was found
                    - artciles may be found in multiple sections,
                        including '' which is the front page
                - text: The text of the document
                    - optionally overrides reading the document from the URI
        """
        doc_ids: List[str] = list(documents.keys())
        docs: List[str] = [None] * len(documents)
        metadata: List[Dict[str, str]] = [None] * len(documents)
        uris: List[str] = [None] * len(documents)

        for i, document in enumerate(documents.values()):
            # read the document content from file if it is a DOCUMENT and we have a uri for it
            # otherwise, get the text from the document object (IE SUMMARYs)
            if document.target == DOCUMENT and uris is not None:
                with open(document.uri, "r", encoding="utf8") as f:
                    docs[i] = f.read()
            else:
                docs[i] = document.text or "TEXT OMMITTED"

            metadata[i] = {
                "collection": self.collection.name,
                "embedding_model": MODEL_NAME,
                **document.dump(),
            }

            # add up to three sections in the metadata for each document
            for section_nbr in range(3):
                metadata[i][f"section{section_nbr}"] = (
                    document.sections[section_nbr]
                    if section_nbr < len(document.sections)
                    else "N/A"
                )
            metadata[i].pop("sections")

            # add up to three headlines in the metadata for each document
            for headline_nbr in range(3):
                metadata[i][f"headline{headline_nbr}"] = (
                    document.headlines[headline_nbr]
                    if headline_nbr < len(document.headlines)
                    else "N/A"
                )
            metadata[i].pop("headlines")

            uris[i] = document.uri

        self.embed_documents(
            document_ids=doc_ids,
            docs=docs,
            uris=uris,
            metadata=metadata,
        )

    def retrieve_documents(
        self, query: str, query_analysis: dict, meta: str, qty_results: str
    ) -> List[str]:
        """
        retrieve_documents
        """

        query_vector = self._embed_model.encode([query]).tolist()[0]
        # Query the chromadb collection and get the set of story URIs
        # that are most similar to the query

        query_type = (
            "story"
            if query_analysis["target"] == DOCUMENT
            else query_analysis["target"]
        )
        section = query_analysis["section"]
        where_clause = {"target": {"$eq": query_type}}
        match len(section):
            case 0:
                pass
            case _:
                where_clause = {
                    "$and": [
                        {"$or": [{f"section{j}": {"$eq": section}} for j in range(3)]},
                        where_clause,
                    ]
                }

        params = {
            "query_embeddings": [query_vector],
            "where": where_clause,
        }
        match qty_results:
            case "ONE":
                params["n_results"] = 1
            case "MANY":
                params["n_results"] = 10
            case "ALL":
                collection_count = self.collection.count()
                params["n_results"] = collection_count
                # do not set a limit
            case _:
                pass

        # print(f"\n QUERY PARAMS: {params}\n")
        embedded_results = self.collection.query(**params)
        print(f"found {len(embedded_results["ids"][0])} results")

        return embedded_results
