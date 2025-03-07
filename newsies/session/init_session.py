"""
nesies.session.init_session
"""

from newsies.chromadb_client import ChromaDBClient
from newsies.llm import LLM as llm
from newsies.session import Session


def init_session(username: str = None) -> tuple[Session, ChromaDBClient]:
    """
    init_session
    """
    chromadb_client = ChromaDBClient()
    session = Session(llm, chromadb_client)
    session.username = username
    return (session, chromadb_client)
