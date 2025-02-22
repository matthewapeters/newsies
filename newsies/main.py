from newsies.ap_news.main import get_latest_news, get_article
from newsies.chromadb_client.main import ChromaDBClient
from newsies.llm import LLM as llm, identify_themes, identify_entities
from newsies.classify import prompt_analysis
from newsies.summarizer import summarize_story
from newsies.chroma_client import CRMADB
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool


def news_summarizer(headlines: dict):
    """
    news_summarizer:
      - Summarize news articles
    """

    def process_story(k, v):
        doc_id = k + "_summary"
        metadata = v.copy()
        metadata["text"] = summarize_story(v["uri"], CRMADB, doc_id)
        metadata["type"] = "summary"
        print(f"Summarized: {k}")
        CRMADB.add_documents({doc_id: metadata})

    with ThreadPoolExecutor(
        max_workers=4
    ) as executor:  # Adjust max_workers based on CPU
        executor.map(lambda kv: process_story(*kv), headlines.items())


def news_loader(headlines: dict) -> dict:
    """
    news_loader:
      - Load news articles
    """
    with Pool(processes=8) as ppool:
        ppool.map(
            get_article,
            [(v["url"], v["headlines"], v["categories"]) for v in headlines.values()],
        )

    CRMADB.add_documents(headlines)


def read_news():
    """
    read_news:
      - read news from the database
    """

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


def main():
    headlines = get_latest_news()
    news_loader(headlines)
    news_summarizer(headlines)
    print("\n" * 5)
    read_news()


if __name__ == "__main__":
    main()
