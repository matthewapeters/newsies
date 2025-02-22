from datetime import datetime
from newsies.chromadb_client.main import ChromaDBClient

COLLECTION = f"ap_news_{datetime.now().strftime(r'%Y-%m-%d')}"
CRMADB = ChromaDBClient()
CRMADB.collection = COLLECTION
CRMADB.language = "en"

TAGS_COLLECTION = "newsies_tags"
TAGSDB = ChromaDBClient()
TAGSDB.collection = TAGS_COLLECTION
TAGSDB.language = "en"
