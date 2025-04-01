"""
newsies.ap_news.article_loader
"""

from multiprocessing import Pool
import os
import pickle
from random import randint
import time
from typing import Dict

from bs4 import BeautifulSoup
import requests

from newsies.redis_client import REDIS
from newsies.document_structures import Document

from .article import Article
from .archive import Archive

# pylint: disable=unidiomatic-typecheck, broad-exception-caught, too-many-arguments

URL = "https://apnews.com"
MAX_TRIES = 5
ARCHIVE = "apnews.com"
os.makedirs(f"./daily_news/{ARCHIVE}", exist_ok=True)


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
                Archive.register_by_publish_date(article)
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
        pikl_path = "./daily_news/apnews.com/latest_news.pkl"
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
