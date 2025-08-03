"""
newsies.session.turn

"""

from typing import List


class Turn:
    """Turn"""

    def __init__(self, **kwargs):
        self.query: str = kwargs.get("query", "")
        self._answer: str = None
        self.articles: List[str] = []

    @property
    def answer(self) -> str:
        """
        answer
            returns the answer to the question
        """
        return self._answer

    @answer.setter
    def answer(self, value: str):
        """
        answer
            sets the answer to the question
        """
        self._answer = value
