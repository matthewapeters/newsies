from datetime import datetime

from newsies.chromadb_client import ChromaDBClient
from newsies.llm import LLM as llm
from newsies.session import Session


def select_collection(archive_date: str = None) -> ChromaDBClient:
    chromadb_client = ChromaDBClient()
    chromadb_client.language = "en"
    if archive_date is None:
        collections = [
            c.replace("ap_news_", "")
            for c in chromadb_client.client.list_collections()
            if c.startswith("ap_news_")
        ]
        collections.sort()
        print("\nExisting News Archives:\n")
        for i in range(0, len(collections), 2):
            if i < len(collections) - 1:
                print(f"{collections[i]}\t\t{collections[i+1]}")
            else:
                print(f"{collections[i]}")
        print("Enter an archive to read (ENTER for today): ", end="")
        archive_date = input()
        if archive_date == "":
            archive_date = datetime.now().strftime(r"%Y-%m-%d")
    collection = f"ap_news_{archive_date}"
    print(f"Reading {collection}\n")
    existing_collections = chromadb_client.client.list_collections()
    if collection in existing_collections:
        chromadb_client.collection_name = collection
        return chromadb_client
    else:
        print(f"collection {collection} does not exist ")
        return select_collection(None)


def read_news(archive_date: str = None):
    """
    read_news:
      - read news from the database
    """
    if archive_date is None:
        chromadb_client = select_collection(archive_date)

    exit_now = False

    session = Session(llm, chromadb_client)

    while exit_now == False:

        # Test locally hosted RAG
        # query = "How does deep learning differ from machine learning and what role does transformer play?"
        query = input(
            f"session {session.id}:\nQuery Newsies Agent {chromadb_client.collection.name} (cc to change collection / quit to exit):\n "
        ).strip()
        match query:
            case "quit":
                exit_now = True
            case "cc":
                prior_collection = chromadb_client.collection.name
                new_chromadb_client = select_collection(None)
                if new_chromadb_client.collection.name != prior_collection:
                    print(
                        f"\nStarting new Session with {new_chromadb_client.collection.name}"
                    )
                    session = Session(llm, new_chromadb_client)
                    chromadb_client = new_chromadb_client
                else:
                    print(
                        f"\n retaining session with {chromadb_client.collection.name}"
                    )
            case "":
                continue
            case _:
                session.query(query)
