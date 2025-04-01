"""
newsies.session
"""

import json
from typing import Dict, List, Union
import uuid

from newsies.session.turn import Turn
from newsies.redis_client import REDIS

# pylint: disable=broad-exception-caught, protected-access, unnecessary-lambda, too-many-instance-attributes, invalid-name


def get_session_params(session_id: str):
    """
    get_session
    """
    raw = REDIS.get(session_id)
    params = json.loads(raw)
    return params


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

    def cache_session(self):
        """
        cache_session
        """
        REDIS.set(self.id, self.toJson())
