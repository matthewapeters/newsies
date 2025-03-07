"""
newsies.session
"""

import uuid
from gpt4all import GPT4All

from newsies.chromadb_client import ChromaDBClient
from newsies.classify import prompt_analysis
from newsies.session.turn import Turn
from newsies import actions

# pylint: disable=broad-exception-caught, protected-access, unnecessary-lambda, too-many-instance-attributes


class Session:
    """
    newsies.session.Session
    """

    def __init__(self, llm, archive_db: ChromaDBClient):
        self.id: uuid.UUID = uuid.uuid4()
        self._llm: GPT4All = llm
        self._archivedb = archive_db
        self._sessiondb = ChromaDBClient()
        self._sessiondb.collection_name = self.id
        self._history = []
        self._context = {}
        self._username: str = None

    @property
    def username(self) -> str:
        """username"""
        return self._username

    @username.setter
    def username(self, name: str):
        self._username = name

    def add(self, t: Turn):
        """
        add
        """
        self._history.append(t)

    @property
    def context(self) -> dict:
        """context"""
        return self._context

    def _clone_last_turn(self, turn: Turn):
        last_turn: Turn = self._history[-1]
        turn.embedded_results = last_turn.embedded_results
        turn.summary_raw = last_turn.summary_raw
        turn.summaries = last_turn.summaries
        turn.response = last_turn.response
        turn._paged_document_map = last_turn._paged_document_map
        turn.target_type = last_turn.target_type

    def query(self, query: str) -> str:
        """query"""
        prompt = ""
        query_analysis = prompt_analysis(query)
        turn = Turn(query, query_analysis)
        # results = None
        print(f"\n(newsies thinks you want to know about {query_analysis})\n")

        match query_analysis["context"]:
            case "NEW":
                embedded_results = self._archivedb.retrieve_documents(
                    query,
                    query_analysis,
                    "text",
                    qty_results=query_analysis["quantity"],
                )

                if len(embedded_results["ids"]) > 0:
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
                    self._clone_last_turn(turn)
            case _:
                self._clone_last_turn(turn)
                match query_analysis["action"]:
                    case actions.READ:
                        # is there an ordinal reference?
                        ordinal = query_analysis["ordinal"][0]
                        target = query_analysis["target"]
                        turn.do_read(
                            ordinal=ordinal, target=target, archivedb=self._archivedb
                        )
                        self.add(turn)
                        return turn.response
                    case actions.LIST:
                        pass
                    case actions.SUMMARIZE:
                        pass
                    case actions.SYNTHESIZE:
                        pass

        prompt: str = turn.prompt()
        print(f"\nPROMPT:\n{prompt}\n-----------------------------------------\n")

        turn.response = self._llm.generate(prompt)
        print("\nRESPONSE:\n", turn.response)
        self.add(turn)
        return turn.response
