"""
newsies.ap_news.latest_news
"""

from datetime import datetime
import time
import os
import re
from typing import Dict
from random import randint
from multiprocessing import Pool
import pickle

import requests
from bs4 import BeautifulSoup

from newsies.targets import DOCUMENT, HEADLINE
from newsies.chroma_client import CRMADB

from ..document_structures import Headline, Document
from .sections import SECTIONS

# pylint: disable=unidiomatic-typecheck

URL = "https://apnews.com"
MAX_TRIES = 5


def path(story_url: str, archive: str = None):
    """path"""
    today = (
        datetime.now().strftime(r"%Y%m%d")
        if archive is None
        else archive.replace("-", "")
    )
    story = story_url.split("/")[-1].split("?")[0]
    os.makedirs(f"./daily_news/{today}", exist_ok=True)
    return f"./daily_news/{today}/{story}.txt"


def get_latest_news() -> Dict[str, Document]:
    """
    get_latest_news
    """

    headlines: Dict[str, Headline] = {}

    for section in SECTIONS:
        print(f"Getting {URL}/{section} news")
        resp = requests.get(f"{URL}/{section}", allow_redirects=True, timeout=5)
        results: bytes = resp.content.decode("utf8")
        # print(results)
        soup = BeautifulSoup(results, features="lxml")
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
        path(url)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_"): Document(
            **{
                "url": url,
                "uri": path(url),
                "date": datetime.now().strftime(r"%Y-%m-%d"),
                "source": "AP News",
                "target": DOCUMENT,
                "headlines": list(
                    set([v.headline for v in headlines.values() if v.url == url])
                ),
                "sections": list(
                    set([v.section for v in headlines.values() if v.url == url])
                ),
            }
        )
        for url in urls
    }
    pickl_path = path("latest_news").replace(".txt", ".pkl")
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
    file_path = path(story_url)
    if os.path.exists(file_path):
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
            article = BeautifulSoup(article_resp.content.decode("utf8"), "html.parser")
            with open(file_path, "w", encoding="utf8") as fh:
                try:
                    for paragraph in article.find(
                        class_="RichTextStoryBody RichTextBody"
                    ).find_all("p"):
                        p = paragraph.text
                        # replace unicode quotes and quote-like characters with apostrophe
                        # https://hexdocs.pm/ex_unicode/Unicode.Category.QuoteMarks.html
                        p = re.sub(r"[\u0018-\u2E42]", "'", p)
                        p += "\n"
                        fh.write(p)
                    if task_status is not None:
                        task_status[task_id] = (
                            f"running: downloaded {doc_id} of {doc_count}"
                        )
                except AttributeError:
                    fh.write("Error parsing article\n")
        case _:
            print(f"Error getting article: {article_resp.status_code}")


def news_loader(
    documents: Dict[str, Document] = None, task_state: dict = None, task_id: str = ""
):
    """
    news_loader:
      - Load news articles
    """
    if documents is None:
        pikl_path = path("latest_news").replace(".txt", ".pkl")
        with open(pikl_path, "rb") as fh:
            documents = pickle.load(fh)

    docs_count = len(documents)
    with Pool(processes=4) as ppool:
        ppool.map(
            download_article,
            [
                (v.url, v.headlines, v.sections, task_state, task_id, i, docs_count)
                for i, v in enumerate(documents.values())
            ],
        )

    CRMADB.add_documents(documents)


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
        pikl_path = path("latest_news").replace(".txt", ".pkl")
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
