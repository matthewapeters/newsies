"""
newsies-clients
Database clients and session management
"""

__version__ = "0.2.0"

# Re-export commonly used items
from .chromadb_client import *
from .redis_client import *
from .session import *

__all__ = [
    "chromadb_client",
    "redis_client", 
    "session",
]
