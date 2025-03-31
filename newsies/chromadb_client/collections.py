"""
newsies.chromadb_client.collections
"""

import json

from newsies.ap_news.archive import get_archive


def collections():
    """
    collections
    """
    archive = get_archive()
    return json.dumps(archive)
