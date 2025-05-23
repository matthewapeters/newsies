"""
newsies.cli.main
"""

from datetime import datetime

from newsies.chromadb_client import ChromaDBClient, collections
from newsies.session import Session, init_session


def select_collection(
    chromadb_client: ChromaDBClient,
    archive_date: str = None,
) -> tuple[Session, ChromaDBClient]:
    """
    select_collection
    """
    prior_collection = (
        chromadb_client.collection.name if chromadb_client.collection else "NONE"
    )
    _collections = collections(chromadb_client, None)

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
    if archive_date in _collections:
        print(f"User Selected Collection: {collection}\n")
    else:
        print(f"collection {archive_date} does not exist")
        return select_collection(chromadb_client, None)

    if collection != prior_collection:
        print(f"\nStarting new Session with {collection}")
        session = Session()
        session.collection = collection
        new_chromadb_client = ChromaDBClient()
        new_chromadb_client.collection_name = collection
        return (session, new_chromadb_client)
    print(f"\n retaining session with {chromadb_client.collection_name}")
    return (None, chromadb_client)


def cli_read_news(one_shot: str = None):
    """
    read_news:
      - read news from the database
    """
    session = init_session()
    # if archive_date is None:
    #    _, chromadb_client = select_collection(chromadb_client, archive_date)

    exit_now = False
    if one_shot:
        resp = session.query(one_shot)
        return f"Response: {resp}"

    while exit_now is False and not one_shot:

        # Test locally hosted RAG
        # query = "How does deep learning differ from machine learning and what
        # role does transformer play?"
        query = input(f"session {session.id}:\nQuery Newsies Agent\n").strip()
        match query:
            case "quit":
                exit_now = True
            case "":
                continue
            case _:
                resp = session.query(query)
                print(f"Response: {resp}")
