"""
newsies.cli.main
"""

from datetime import datetime

from newsies.chromadb_client import ChromaDBClient
from newsies.llm import LLM as llm
from newsies.session import Session, init_session


def select_collection(
    chromadb_client: ChromaDBClient,
    archive_date: str = None,
) -> tuple[Session, ChromaDBClient]:
    """
    select_collection
    """
    prior_collection = chromadb_client.collection.name
    _collections = select_collection(chromadb_client, None)

    for i in range(0, len(_collections), 2):
        if i < len(_collections) - 1:
            print(f"{_collections[i]}\t\t{_collections[i+1]}")
        else:
            print(f"{_collections[i]}")
    print("Enter an archive to read (ENTER for today): ", end="")
    archive_date = input()
    if archive_date == "":
        archive_date = datetime.now().strftime(r"%Y-%m-%d")

    collection = f"ap_news_{archive_date}"
    if collection in _collections:
        print(f"Reading {collection}\n")
        chromadb_client.collection_name = collection
    else:
        print(f"collection {collection} does not exist ")
        return select_collection(chromadb_client, None)

    if chromadb_client.collection.name != prior_collection:
        print(f"\nStarting new Session with {chromadb_client.collection.name}")
        new_chromadb_client = ChromaDBClient()
        session = Session(llm, new_chromadb_client)
        return (session, new_chromadb_client)
    print(f"\n retaining session with {chromadb_client.collection.name}")
    return (None, chromadb_client)


def collections(
    chromadb_client: ChromaDBClient,
    archive_date: str = None,
):
    """
    collections
    """
    _collections = [f"ap_news_{archive_date}"]
    if archive_date is None:
        _collections = [
            c.replace("ap_news_", "")
            for c in chromadb_client.client.list_collections()
            if c.startswith("ap_news_")
        ]
        _collections.sort()
    return _collections


def cli_read_news(archive_date: str = None):
    """
    read_news:
      - read news from the database
    """
    session, chromadb_client = init_session(None, None, None)
    if archive_date is None:
        _, chromadb_client = select_collection(chromadb_client, archive_date)

    exit_now = False

    while exit_now is False:

        # Test locally hosted RAG
        # query = "How does deep learning differ from machine learning and what
        # role does transformer play?"
        query = input(
            f"session {session.id}:\nQuery Newsies Agent "
            f"{chromadb_client.collection.name} (cc to change collection / quit to exit):\n "
        ).strip()
        match query:
            case "quit":
                exit_now = True
            case "cc":
                new_session, chromadb_client = select_collection(
                    chromadb_client, archive_date
                )
                if new_session:
                    session = new_session
            case "":
                continue
            case _:
                session.query(query)
