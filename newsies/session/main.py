"""
newsies.session
"""

import uuid
from gpt4all import GPT4All

from newsies.chromadb_client import ChromaDBClient
from newsies.classify import prompt_analysis


class Session:
    """
    newsies.session.Session
    """

    def __init__(self, llm, archive_db: ChromaDBClient):
        self.id: uuid.UUID = uuid.uuid4()
        self._llm: GPT4All = llm
        self._archivedb = archive_db
        self._sessiondb = ChromaDBClient()
        self._sessiondb.collection = self.id
        self.history = []
        self._context = {}

    def _add(self, label: str, o: object):
        self._history.append({label: o})

    @property
    def context(self) -> dict:
        return self._context

    def query(self, query: str):
        self._add("query", query)
        query_analysis = prompt_analysis(query)
        self._add("request_meta", query_analysis)
        print(f"(newsies thinks you want to know about {query_analysis})")

        # how many documents to query?  probably need to move this to query_analysis
        results = 5

        # analyze request

        embedded_results = self._archivedb.retrieve_documents(
            query, query_analysis, "text", results
        )

        if len(embedded_results["ids"]) > 0:
            self._add("embedded_results", embedded_results)

            summary_raw = self.collection.get(
                ids=embedded_results["ids"][0], include=["documents"]
            )

            self._add("summary_raw", summary_raw)

            summaries = {
                summary_raw["ids"][i]: summary_raw["documents"][i]
                for i in range(len(summary_raw["ids"]))
            }

            self._add("summaries", summaries)

            result_count = len(embedded_results["ids"][0])
            for i in range(result_count):
                context += f"document {i}: {embedded_results['ids'][0][i]}\n"
                context += f"uri: {embedded_results['metadatas'][0][i]['uri']}\n"
                context += f"title(s): {embedded_results['metadatas'][0][i]['headlines']}\n"
                context += f"summary: {summaries[embedded_results['ids'][0][i]]}\n"

            prompt = f"Context: {{\n{context}\n}}\nQuestion: {query}\nAnswer:"

        response = self._llm.generate(prompt)

        # response needs to be added to running context
        self._add("response", response)

        print("\nRESPONSE:\n", response)
,