"""
newsies.session
"""

import json
from typing import Dict, List, Union
import uuid

from newsies.llm import LLM as llm
from newsies.chromadb_client import ChromaDBClient, get_all_headlines
from newsies.classify import prompt_analysis
from newsies.session.turn import Turn
from newsies import actions
from newsies import targets

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

    def query(self, query: str = None, query_analysis: Dict = None) -> str:
        """query"""
        prompt = ""
        if query_analysis is None and query is not None:
            query_analysis = prompt_analysis(query)

        turn = Turn(query=query, query_analysis=query_analysis)
        # results = None
        print(f"\n(newsies thinks you want to know about {query_analysis})\n")
        _archivedb = ChromaDBClient()
        _archivedb.collection_name = self._collection
        match [
            query_analysis["context"],
            query_analysis["target"],
            query_analysis["quantity"],
        ]:
            # These scenarios refer to a new query
            case ["NEW", targets.HEADLINE, "ALL"]:
                headlines = get_all_headlines(
                    self.collection, query_analysis["section"]
                )

                output = {
                    "section": query_analysis["section"],
                    "headlines": headlines,
                }
                turn.paged_document_map = [{"0": output}]
                self.add_history(turn)
                return output
            case ["NEW", targets.HEADLINE, "MANY"]:
                # as opposed to all - are these paged or filtered somehow
                headlines = get_all_headlines(
                    self.collection, query_analysis["section"]
                )

                output = {
                    "section": query_analysis["section"],
                    "headlines": headlines,
                }
                turn.paged_document_map = [{"0": output}]
                self.add_history(turn)
                return output
            case ["NEW", targets.HEADLINE, "ONE"]:
                # how can we ask a new query about a headline we know nothing about?
                pass
            case ["NEW", targets.DOCUMENT, "ALL"]:
                pass
            case ["NEW", targets.DOCUMENT, "MANY"]:
                pass
            case ["NEW", targets.DOCUMENT, "ONE"]:
                pass
            case ["NEW", targets.SUMMARY, "ALL"]:
                pass
            case ["NEW", targets.SUMMARY, "MANY"]:
                pass
            case ["NEW", targets.SUMMARY, "ONE"]:
                pass

            # These scenarios refer to a prior query
            case ["OLD", targets.HEADLINE, "ALL"]:
                pass
            case ["OLD", targets.HEADLINE, "MANY"]:
                pass
            case ["OLD", targets.HEADLINE, "ONE"]:
                pass
            case ["OLD", targets.DOCUMENT, "ALL"]:
                pass
            case ["OLD", targets.DOCUMENT, "MANY"]:
                pass
            case ["OLD", targets.DOCUMENT, "ONE"]:
                pass
            case ["OLD", targets.SUMMARY, "ALL"]:
                pass
            case ["OLD", targets.SUMMARY, "MANY"]:
                pass
            case ["OLD", targets.SUMMARY, "ONE"]:
                pass
            case _:
                pass

        prompt: str = turn.get_prompt()
        turn.response = llm.generate(prompt)
        self.add(turn)
        return turn.response
