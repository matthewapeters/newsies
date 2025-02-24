"""
newsies.session
"""

import uuid
from gpt4all import GPT4All

from newsies.chromadb_client import ChromaDBClient
from newsies.classify import prompt_analysis


class Turn:
    def __init__(self, query: str, query_analysis: dict):
        self.query = query
        self.query_analysis = query_analysis
        self._embedded_results = None
        self._summary_raw = None
        self._summaries = None
        self._prompt = None
        self._response = None

    @property
    def embedded_results(self):
        return self._embedded_results

    @embedded_results.setter
    def embedded_results(self, er):
        self._embedded_results = er

    @property
    def summary_raw(self):
        return self._summary_raw

    @summary_raw.setter
    def summary_raw(self, sr):
        self._summary_raw = sr

    @property
    def summaries(self):
        return self._summaries

    @summaries.setter
    def summaries(self, s):
        self._summaries = s

    def prompt(self, include_summaries: bool = True, include_uri=True) -> str:
        context = ""
        result_count = len(self.embedded_results["ids"][0])
        context = f"{self.query_analysis["quantity"]} story[s] from categies:{self.query_analysis["categories"]}\n"
        for i in range(result_count):
            context += f"document {i}: {self.embedded_results['ids'][0][i]}\n"
            if include_uri:
                context += (
                    f"uri {i}: {self.embedded_results['metadatas'][0][i]['uri']}\n"
                )
            context += (
                f"title(s): {self.embedded_results['metadatas'][0][i]['headlines']}\n"
            )
            if include_summaries:
                context += (
                    f"summary: {self.summaries[self.embedded_results['ids'][0][i]]}\n"
                )

        context += (
            "" if self.response is None else f"Agent Last Response: {self.response}\n"
        )

        self._prompt = (
            f"Context: {{\n{context}\n}}\nUser Question: {self.query}\nAgent Answer:"
        )

        if len(self._prompt) > 2048 and include_summaries:
            print("\nremoving summaries from context\n")
            self.prompt(include_summaries=False)
        if len(self._prompt) > 2048 and include_uri:
            print("\nremoving uris from context\n")
            self.prompt(include_summaries=False, include_uri=False)
        return self._prompt

    @property
    def response(self):
        return self._response

    @response.setter
    def response(self, r):
        self._response = r


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
        self._history = []
        self._context = {}

    def add(self, t: Turn):
        self._history.append(t)

    @property
    def context(self) -> dict:
        return self._context

    def query(self, query: str):
        prompt = ""
        query_analysis = prompt_analysis(query)
        turn = Turn(query, query_analysis)
        results = None
        print(f"\n(newsies thinks you want to know about {query_analysis})\n")

        if query_analysis["context"] == "NEW":
            embedded_results = self._archivedb.retrieve_documents(
                query, query_analysis, "text", qty_results=query_analysis["quantity"]
            )

            if len(embedded_results["ids"]) > 0:
                print(embedded_results["ids"])
                try:
                    turn.embedded_results = embedded_results

                    summary_raw = self._archivedb.collection.get(
                        ids=embedded_results["ids"][0], include=["documents"]
                    )
                    turn.summary_raw = summary_raw

                    summaries = {
                        summary_raw["ids"][i]: summary_raw["documents"][i]
                        for i in range(len(summary_raw["ids"]))
                    }

                    turn.summaries = summaries
                except Exception as e:
                    print(f"ERROR: {e}")
                    print("Try re-phrasing the question")
                    return
            else:
                last_turn: Turn = self._history[-1]
                turn.embedded_results = last_turn.embedded_results
                turn.summary_raw = last_turn.summary_raw
                turn.summaries = last_turn.summaries
                turn.response = last_turn.response
        else:
            last_turn = self._history[-1]
            turn.embedded_results = last_turn.embedded_results
            turn.summary_raw = last_turn.summary_raw
            turn.summaries = last_turn.summaries
            turn.response = last_turn.response

        prompt = turn.prompt()
        print(f"\nPROMPT:\n{prompt}\n-----------------------------------------\n")

        turn.response = self._llm.generate(prompt)
        print("\nRESPONSE:\n", turn.response)
        self.add(turn)
