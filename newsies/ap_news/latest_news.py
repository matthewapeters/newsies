"""
newsies.ap_news.latest_news
"""

from datetime import datetime
from typing import Dict
import os
import pickle
import time

from bs4 import BeautifulSoup
import requests

from newsies.targets import DOCUMENT

# from newsies.chroma_client import CRMADB
from newsies.document_structures import Headline, Document

from .sections import SECTIONS

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
