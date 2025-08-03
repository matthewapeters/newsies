"""
newsies.chromadb_client.collections
"""

from .main import ChromaDBClient


def collections(
    chromadb_client: ChromaDBClient,
    archive_date: str = None,
):
    """
    collections
    """
    _collections = [archive_date]
    if archive_date is None:
        _collections = [
            c.replace("ap_news_", "")
            for c in chromadb_client.client.list_collections()
            if c.startswith("ap_news_")
        ]
        _collections.sort()
    return _collections
