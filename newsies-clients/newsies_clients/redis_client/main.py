"""
newsies.redis_client.main
"""

import os
import redis

# Handle missing environment variables gracefully
try:
    user, passwd = os.environ["REDIS_CREDS"].split(":")
except KeyError:
    # Default values for development/testing
    user, passwd = "default", "password"

REDIS = redis.Redis(host="localhost", db=0, port=6379, password=passwd, username=user)
