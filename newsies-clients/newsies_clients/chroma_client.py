"""
newsies.chroma_client
"""

from datetime import datetime
from newsies_clients.chromadb_client.main import ChromaDBClient

CRMADB = ChromaDBClient()
CRMADB.language = "en"
# collection can be changed per session
COLLECTION = f"ap_news_{datetime.now().strftime(r'%Y-%m-%d')}"
CRMADB.collection_name = COLLECTION
