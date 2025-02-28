from datetime import datetime
from newsies.chromadb_client import ChromaDBClient
from newsies.classify import TAGS_COLLECTION

CRMADB = ChromaDBClient()
CRMADB.language = "en"
# collection can be changed per session
COLLECTION = f"ap_news_{datetime.now().strftime(r'%Y-%m-%d')}"
CRMADB.collection_name = COLLECTION
