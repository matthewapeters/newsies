"""
newsies.session
"""

import json
from typing import Any, Dict, List, Union
import uuid

from newsies.llm import LLM as llm
from newsies.chromadb_client import ChromaDBClient
from newsies.classify import prompt_analysis
from newsies.session.turn import Turn
from newsies import actions

# pylint: disable=broad-exception-caught, protected-access, unnecessary-lambda, too-many-instance-attributes, invalid-name


class Session:
    """
    newsies.session.Session
    """

    def __init__(self, **kwargs):
        self.id: str = kwargs.get("id", str(uuid.uuid4()))
        h = kwargs.get("history")
        history = None
        match str(type(h)):
            case "<class 'str'>":
                history = [Turn(**i) for i in json.loads(h) if isinstance(i, dict)]
            case "<class 'list'>":
                history = [Turn(**i) for i in h if isinstance(i, dict)]
            case _:
                history = []
        self._history: List[Turn] = history
        self._context: Dict = kwargs.get("context", {})
        self._username: str = kwargs.get("username")
        self._collection: str = kwargs.get("collection")

    def toJson(self):
        """
        toJson
            returns a JSON string of the session
            Used in caching
        """
        return json.dumps(self.dump())

    def dump(self) -> Dict[str, Union[str, List, Dict]]:
        """
        dump
            returns a dict of the session
        """
        return {
            "id": self.id,
            "history": [t.__dict__ for t in self._history],
            "context": self._context,
            "username": self._username,
            "collection": self._collection,
        }

    def add_history(self, o):
        """add_history"""
        self._history.append(o)

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

    @property
    def collection(self):
        """collection"""
        return self._collection

    @collection.setter
    def collection(self, c: str):
        self._collection = c

    def _clone_last_turn(self, turn: Turn):
        if len(self._history) == 0:
            return
        last_turn: Turn = self._history[-1]
        turn.embedded_results = last_turn.embedded_results
        turn.summary_raw = last_turn.summary_raw
        turn.summaries = last_turn.summaries
        turn.response = last_turn.response
        turn.paged_document_map = last_turn.paged_document_map
        turn.target_type = last_turn.target_type

    def query(self, query: str) -> str:
        """query"""
        prompt = ""
        query_analysis = prompt_analysis(query)
        turn = Turn(query=query, query_analysis=query_analysis)
        # results = None
        print(f"\n(newsies thinks you want to know about {query_analysis})\n")
        _archivedb = ChromaDBClient()
        _archivedb.collection_name = self._collection
        match query_analysis["context"]:
            case "NEW":
                embedded_results = _archivedb.retrieve_documents(
                    query,
                    query_analysis,
                    "text",
                    qty_results=query_analysis["quantity"],
                )

                if len(embedded_results["ids"]) > 0:
                    try:
                        turn.embedded_results = embedded_results

                        summary_raw = _archivedb.collection.get(
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
                            ordinal=ordinal, target=target, archivedb=_archivedb
                        )
                        self.add(turn)
                        return turn.response
                    case actions.LIST:
                        pass
                    case actions.SUMMARIZE:
                        pass
                    case actions.SYNTHESIZE:
                        pass

        prompt: str = turn.get_prompt()
        print(f"\nPROMPT:\n{prompt}\n-----------------------------------------\n")

        turn.response = llm.generate(prompt)
        print("\nRESPONSE:\n", turn.response)
        self.add(turn)
        return turn.response
