"""
newsies.redis_client.main
"""

import os
import redis

user, passwd = os.environ["REDIS_CREDS"].split(":")

REDIS = redis.Redis(host="localhost", db=0, port=6379, password=passwd, username=user)
