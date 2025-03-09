"""
newsies.redis_client.main
"""

import json
import os
import redis
from newsies.session import Session

user, passwd = os.environ["REDIS_CREDS"].split(":")

REDIS = redis.Redis(host="localhost", db=0, port=6379, password=passwd, username=user)


def cache_session(session: Session):
    """
    cache_session
    """
    REDIS.set(session.id, session.toJson())


def get_session(session_id: str):
    """
    get_session
    """
    raw = REDIS.get(session_id)
    params = json.loads(raw)
    return Session(**params)
