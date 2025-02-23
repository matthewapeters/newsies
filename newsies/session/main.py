import uuid
from newsies.chromadb_client import ChromaDBClient


class Session:
    def __init__(self):
        self.id: uuid.UUID = uuid.uuid4()
        self._db = ChromaDBClient()
        self._db.collection = self.id
        self.history = []
        self.context = {}
