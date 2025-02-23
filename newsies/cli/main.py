from newsies.chroma_client import CRMADB
from newsies.classify import prompt_analysis
from newsies.llm import LLM as llm


def read_news(archive_date: str):
    """
    read_news:
      - read news from the database
    """
    collection = f"ap_news_{archive_date}"
    existing_collections = CRMADB.client.list_collections()
    print(existing_collections)
    if collection in existing_collections:
        CRMADB.collection = collection
    else:
        print(f"collection {collection} does not exist - run newsies get-news")
        return

    exit_now = False
    while exit_now == False:

        # Test locally hosted RAG
        # query = "How does deep learning differ from machine learning and what role does transformer play?"
        query = input("Question (quit to exit): ").strip()
        if query == "quit":
            exit_now = True
        elif query == "":
            continue
        else:
            # analyze request
            request_meta = prompt_analysis(query)
            print(f"(newsies thinks you want to know about {request_meta})")
            # Generate response using GPT4All
            response = CRMADB.generate_rag_response(query.lower(), request_meta, llm, 5)
            print("\nRESPONSE:\n", response)
