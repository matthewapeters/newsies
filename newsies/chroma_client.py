from datetime import datetime
from newsies.chromadb_client.main import ChromaDBClient

CRMADB = ChromaDBClient()
CRMADB.language = "en"
# collection can be changed per session
COLLECTION = f"ap_news_{datetime.now().strftime(r'%Y-%m-%d')}"
CRMADB.collection = COLLECTION

TAGS_COLLECTION = "newsies_tags"
TAGSDB = ChromaDBClient()
TAGSDB.collection = TAGS_COLLECTION
TAGSDB.language = "en"
