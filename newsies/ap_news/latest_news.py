"""
newsies.ap_news.latest_news
"""

from datetime import datetime
import time
import os
from typing import Dict
from random import randint
from multiprocessing import Pool
import pickle

import requests
from bs4 import BeautifulSoup

from newsies.targets import DOCUMENT, HEADLINE
from newsies.chroma_client import CRMADB
from newsies.redis_client.main import REDIS

from ..document_structures import Headline, Document
from .sections import SECTIONS
from .article import Article
from .named_entitiy_visitor import NamedEntityVisistor
from .embedding_visitor import EmbeddingVisitor

# pylint: disable=unidiomatic-typecheck, broad-exception-caught

URL = "https://apnews.com"
MAX_TRIES = 5
ARCHIVE = "apnews.com"
os.makedirs(f"./daily_news/{ARCHIVE}", exist_ok=True)


def path(file: str):
    """path"""
    return f"./daily_news/{ARCHIVE}/{file}"


def get_latest_news() -> Dict[str, Document]:
    """
    get_latest_news
        if latest_news.pkl is less than one hour old, retuns the pickled
        documents
        otherwise, scrapes the AP News site for the latest news links
        and pickles the documents for downstream use
    """
    if (
        os.path.exists(path("latest_news.pkl"))
        and time.time() - os.path.getmtime(path("latest_news.pkl")) < 3600
    ):
        print("Using cached news")
        with open(path("latest_news.pkl"), "rb") as fh:
            return pickle.load(fh)

    headlines: Dict[str, Headline] = {}

    for section in SECTIONS:
        print(f"Getting {URL}/{section} news")
        resp = requests.get(f"{URL}/{section}", allow_redirects=True, timeout=5)
        results: bytes = resp.content.decode("utf8")
        # print(results)
        soup = BeautifulSoup(results, features="html.parser")
        if type(soup) == "NoneType":
            continue
        items = soup.find_all("a")
        print(f"\tFound {len(items)} links")
        raw_headlines = []
        for link in items:
            if (
                "href" in link.attrs
                and f"{URL}/article/" in link.attrs["href"]
                and len(link.text.strip()) > 1
            ):
                raw_headlines.append(link)

        raw_headlines.sort(key=lambda x: x.text)
        # de-dupe headlines
        headline_list = set(raw_headlines)
        print(f"\tFound {len(headline_list)} stories in links")

        for s in headline_list:
            headlines[f"{section}: {s.text}"] = Headline(
                **{
                    "url": s.attrs["href"],
                    "headline": s.text,
                    "section": section,
                }
            )

    urls = list(set([v.url for v in headlines.values()]))

    documents: Dict[str, Document] = {
        url: Document(
            **{
                "url": url,
                "date": datetime.now().strftime(r"%Y-%m-%d"),
                "source": "AP News",
                "target": DOCUMENT,
                "headlines": list(
                    [v.headline for v in headlines.values() if v.url == url]
                ),
                "sections": list(
                    [v.section for v in headlines.values() if v.url == url]
                ),
            }
        )
        for url in urls
    }
    pickl_path = path("latest_news.pkl")
    with open(pickl_path, "wb") as fh:
        pickle.dump(documents, fh)
    return documents


def download_article(
    work: tuple,
    backoff: int = 0,
    tries: int = 0,
    task_status: dict = None,
    task_id: str = "",
    doc_id: int = 0,
    doc_count: int = 0,
):
    """
    download_article

    get the article from the story_url and write it to a file

    * if a file for the story already exists, return without doing anything
    * if too many retries have been attempted, return without doing anything
    * As this method is commonly called in a process pool, we will sleep a
        short random time before initially trying to get the article
    * If we get a 429 status code, we will sleep for a random time between
        5 and 30 seconds before trying again
        * On repeated 428 status code, the sleep will increase between the
            last backoff time and 30 seconds more
    * If we get a 200 status code, we will write the article to a file

    :param work: tuple
        story_url: str - the URL of the story
        headlines: List[str] - a list of headlines for the story
        sections: List[str] - a list of sections for the story
    :param backoff: int - the number of seconds to wait before retrying
    :param tries: int - the number of times this function has been called
    :param task_status: dict - pipeline task status dict
    :param task_id: str - pipeline task id
    :param doc_id: int - enumeration index of the current document
    :param doc_count: int - total count of documents in the task
    """

    story_url, headlines, sections = work

    if tries > MAX_TRIES:
        return

    cached_story_path = REDIS.get(story_url)
    if cached_story_path is not None:
        return
    if backoff == 0:
        backoff = randint(1, 5)
        time.sleep(backoff)

    article_resp = requests.get(story_url, allow_redirects=True, timeout=5)

    match article_resp.status_code:
        case 429:
            # Too many requests - pick a random backoff time
            backoff = randint(max(backoff, 5), 30 + backoff)
            print(
                f"Too many requests, sleeping for {backoff} seconds ({tries + 1} try)"
            )
            time.sleep(backoff)
            download_article(
                (story_url, headlines, sections), backoff=backoff, tries=tries + 1
            )
            return
        case 200:
            try:
                article = Article(
                    archive=ARCHIVE,
                    url=story_url,
                    bs=BeautifulSoup(
                        article_resp.content.decode("utf8"), "html.parser"
                    ),
                )
                for i, headline in enumerate(headlines):
                    s = sections[i]
                    article.section_headlines[s] = headline
                article.pickle()
                article.cache()
                if task_status is not None:
                    task_status[task_id] = (
                        f"running: downloaded {doc_id} of {doc_count}"
                    )
            except Exception as e:
                print(f"Error processing article: {e}")
        case _:
            print(f"Error getting article: {article_resp.status_code}")


def article_loader(
    documents: Dict[str, Document] = None, task_state: dict = None, task_id: str = ""
):
    """
    news_loader:
      - Load news articles
    """
    if documents is None:
        pikl_path = path("latest_news.pkl")
        with open(pikl_path, "rb") as fh:
            documents = pickle.load(fh)

    docs_count = len(documents)
    with Pool(processes=4) as ppool:
        ppool.starmap(
            download_article,
            [
                (
                    (v.url, v.headlines, v.sections),  # work
                    0,  # backoff
                    0,  # tries
                    task_state,  # task_status
                    task_id,  # task_id
                    i,  # doc_id
                    docs_count,  # doc_count
                )
                for i, v in enumerate(documents.values())
            ],
        )


def detect_named_entities(
    story_url: str,
    task_status: dict = None,
    task_id: str = "",
    doc_id: int = 0,
    doc_count: int = 0,
):
    """
    detect_named_entities
    """
    uri = REDIS.get(story_url)
    with open(uri, "rb") as fh:
        article = pickle.load(fh)

    v = NamedEntityVisistor()
    v.visit(article)

    article.pickle()
    if task_status is not None:
        task_status[task_id] = f"running: NER {doc_id} of {doc_count}"


def article_ner(
    documents: Dict[str, Document] = None, task_state: dict = None, task_id: str = ""
):
    """
    article_ner
    """
    if documents is None:
        pikl_path = path("latest_news.pkl")
        with open(pikl_path, "rb") as fh:
            documents = pickle.load(fh)

    docs_count = len(documents)
    with Pool(processes=4) as ppool:
        ppool.starmap(
            detect_named_entities,
            [
                (
                    v.url,
                    task_state,  # task_status
                    task_id,  # task_id
                    i,  # doc_id
                    docs_count,  # doc_count
                )
                for i, v in enumerate(documents.values())
            ],
        )


def generate_embeddings(
    story_url: str,
    task_status: dict = None,
    task_id: str = "",
    doc_id: int = 0,
    doc_count: int = 0,
):
    """
    generate_embeddings
    """
    uri = REDIS.get(story_url)
    with open(uri, "rb") as fh:
        article = pickle.load(fh)

    v = EmbeddingVisitor()
    v.visit(article)
    article.pickle()
    if task_status is not None:
        task_status[task_id] = f"running: embeddings {doc_id} of {doc_count}"


def article_embeddings(
    documents: Dict[str, Document] = None, task_state: dict = None, task_id: str = ""
):  # pylint: disable=unused-argument
    """
    article_embeddings
    """
    if documents is None:
        pikl_path = path("latest_news.pkl")
        with open(pikl_path, "rb") as fh:
            documents = pickle.load(fh)

    with Pool(processes=4) as ppool:
        ppool.starmap(
            generate_embeddings,
            [
                (
                    v.url,
                    task_state,  # task_status
                    task_id,  # task_id
                    i,  # doc_id
                    len(documents),  # doc_count
                )
                for i, v in enumerate(documents.values())
            ],
        )


def news_loader(
    documents: Dict[str, Document] = None, task_state: dict = None, task_id: str = ""
):
    """
    news_loader:
      - Load news articles
    """
    if documents is None:
        pikl_path = path("latest_news.pkl")
        with open(pikl_path, "rb") as fh:
            documents = pickle.load(fh)

    docs_count = len(documents)
    with Pool(processes=4) as ppool:
        ppool.starmap(
            download_article,
            [
                (
                    (v.url, v.headlines, v.sections),
                    0,
                    0,
                    task_state,
                    task_id,
                    i,
                    docs_count,
                )
                for i, v in enumerate(documents.values())
            ],
        )

    # CRMADB.add_documents(documents)


def headline_loader(
    documents: Dict[str, Document] = None, task_state: dict = None, task_id: str = ""
):
    """
    headline_loader
    :param documents: dict
    :param task_state: dict - pipeline task status
    :param task_id: str - pipeline task id
    """
    if task_state is not None:
        task_state[task_id] = f"runinng - loading {len(documents)} headlines"
    if documents is None:
        pikl_path = path("latest_news.pkl")
        with open(pikl_path, "rb") as fh:
            documents = pickle.load(fh)

    headlines: Dict[str, Document] = {}
    for doc_id, doc in documents.items():
        for headline_idx, headline in enumerate(doc.headlines):
            h_doc = Document(**doc.dump())
            h_doc.text = headline
            h_doc.headlines = [headline]
            h_doc.target = HEADLINE
            headlines[f"{doc_id}_hl{headline_idx}"] = h_doc

    CRMADB.add_documents(headlines)
    if task_state is not None:
        task_state[task_id] = f"runinng - {len(documents)} headlines loaded"
