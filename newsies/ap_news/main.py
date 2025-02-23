"""
newsies.ap_news.main
"""

from bs4 import BeautifulSoup
from datetime import datetime
import os
from random import randint
import requests
import time


MAX_TRIES = 5
URL = "https://apnews.com"
PAGES = [
    "",
    "world-news",
    "us-news",
    "politics",
    "business",
    "technology",
    "science",
    "health",
    "entertainment",
    "sports",
    "oddities",
]


def get_latest_news():

    headlines = {}

    for page in PAGES:
        print(f"Getting {URL}/{page} news")
        resp = requests.get(f"{URL}/{page}", allow_redirects=True)
        results: bytes = resp.content.decode("utf8")
        # print(results)
        soup = BeautifulSoup(results, features="lxml")
        if type(soup) == "NoneType":
            continue
        items = soup.find_all("a")
        print(f"\tFound {len(items)} links")
        stories = []
        for link in items:
            if (
                "href" in link.attrs
                and f"{URL}/article/" in link.attrs["href"]
                and len(link.text.strip()) > 1
            ):
                stories.append(link)

        stories.sort(key=lambda x: x.text)
        stories = set(stories)
        print(f"\tFound {len(stories)} stories in links")
        for s in stories:
            headlines[s.text] = {
                "url": s.attrs["href"],
                "path": path(s.attrs["href"]),
                "date": datetime.now().isoformat(),
                "source": "AP News",
                "category": page,
            }

    urls = list(set([v["url"] for v in headlines.values()]))

    documents = {
        path(url)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_"): {
            "url": url,
            "uri": path(url),
            "type": "story",
            "date": datetime.now().strftime(r"%Y-%m-%d"),
            "source": "AP News",
            "headlines": "'"
            + "','".join(
                list(set([k for k, v in headlines.items() if v["url"] == url]))
            )
            + "'",
            "categories": ",".join(
                list(
                    set([v["category"] for v in headlines.values() if v["url"] == url])
                )
            ),
        }
        for url in urls
    }

    return documents


def path(story_url: str):
    today = datetime.now().strftime(r"%Y%m%d")
    story = story_url.split("/")[-1].split("?")[0]
    os.makedirs(f"./daily_news/{today}", exist_ok=True)
    return f"./daily_news/{today}/{story}.txt"


def get_article(
    work: tuple,
    backoff: int = 0,
    tries: int = 0,
):
    """
    get_article

    get the article from the story_url and write it to a file

    * if a file for the story already exists, return without doing anything
    * if too many retries have been attempted, return without doing anything
    * As this method is commonly called in a process pool, we will sleep a short random time before initially trying to get the article
    * If we get a 429 status code, we will sleep for a random time between 5 and 30 seconds before trying again
        * On repeated 428 status code, the sleep will increase between the last backoff time and 30 seconds more
    * If we get a 200 status code, we will write the article to a file

    :param work: tuple
        story_url: str - the URL of the story
        headlines: List[str] - a list of headlines for the story
        categories: List[str] - a list of categories for the story
    :param backoff: int - the number of seconds to wait before retrying
    :param tries: int - the number of times this function has been called
    """

    story_url, headlines, categories = work

    if tries > MAX_TRIES:
        return
    file_path = path(story_url)
    if os.path.exists(file_path):
        return
    if backoff == 0:
        backoff = randint(1, 5)
        time.sleep(backoff)

    article_resp = requests.get(story_url, allow_redirects=True)

    match article_resp.status_code:
        case 429:
            # Too many requests - pick a random backoff time
            backoff = randint(max(backoff, 5), 30 + backoff)
            print(
                f"Too many requests, sleeping for {backoff} seconds ({tries + 1} try)"
            )
            time.sleep(backoff)
            get_article(
                (story_url, headlines, categories), backoff=backoff, tries=tries + 1
            )
            return
        case 200:
            article = BeautifulSoup(article_resp.content.decode("utf8"), "html.parser")
            with open(file_path, "w", encoding="utf8") as fh:
                try:
                    for paragraph in article.find(
                        class_="RichTextStoryBody RichTextBody"
                    ).find_all("p"):
                        fh.write(f"{paragraph.text}\n")
                except AttributeError:
                    fh.write(f"Error parsing article\n")
        case _:
            print(f"Error getting article: {article_resp.status_code}")
