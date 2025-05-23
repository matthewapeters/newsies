"""
nesies.session.init_session
"""

from newsies.session import Session


def init_session(username: str = None) -> Session:
    """
    init_session
    """
    session = Session()
    session.username = username
    return session
