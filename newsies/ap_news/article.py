"""
newsies.ap_news.article
"""

from datetime import datetime
from typing import Any, List, Dict
import json
import os
import pickle
import re

from bs4 import BeautifulSoup
import requests

from newsies.redis_client import REDIS
from newsies.targets import ARTICLE


class Article:
    """Article"""

    type = ARTICLE

    def __init__(self, archive: str, url: str, bs: BeautifulSoup = None):
        self.archive: str = archive
        self.url: str = url
        if bs is None:
            resp = requests.get(url, allow_redirects=True, timeout=5)
            results: bytes = resp.content.decode("utf8")
            self.bs = BeautifulSoup(results, features="lxml")
        else:
            self.bs = bs
        # these are set by processing the story
        self.summary: str = ""
        self.named_entities: List[str] = []
        self.embeddings: List[float] = []
        self.section_headlines: Dict[str, str] = {}
        self.metas: List[Any] = self.bs.find_all("meta")
        # the following are set by the methods below
        self.permutive_data: Dict[str, Any] = self._get_permutive_data()
        self.story: str = self._get_story()
        self.item_id: str = self.permutive_data.get("item_id", "")
        self.publish_date: datetime = self._get_publish_date()
        self.keywords: List[str] = self._get_keywords()
        self.open_graph: Dict[str, str] = self._get_open_graph()
        self.authors: List[str] = self._get_authors()

    def _get_permutive_data(self) -> Dict[str, Any]:
        """
        _get_permutive_data
        """
        pdl = [
            json.loads(m["content"])
            for m in self.metas
            if m.get("name") == "permutive-dataLayer"
        ][0]
        if "category" in pdl and "headline" in pdl:
            self.section_headlines[pdl["category"]] = pdl["headline"]
        return pdl

    def _get_keywords(self) -> List[str]:
        """
        _get_keywords
        """
        keyword_meta_properties = ["keywords", "ntv-kv"]
        keywords = [
            kw.strip()
            for m in self.metas
            if m.get("name", "unknown") in keyword_meta_properties
            for kw in (m.get("content") or m.get("values")).split(",")
        ]
        keywords.extend(self.permutive_data.get("tags", []))
        self.named_entities.extend(keywords)
        return list(set(keywords))

    def _get_publish_date(self) -> datetime:
        """
        _get_publish_date
        """
        return (
            datetime.fromisoformat(self.permutive_data["publication_date"])
            or [
                datetime.fromisoformat(mm.get("content"))
                for mm in self.metas
                if mm.get("property", "unknown").startswith("article:published_time")
            ][0]
        )

    def _get_open_graph(self) -> Dict[str, str]:
        """
        _get_open_graph
        """
        return {
            mm.get("property"): mm.get("content")
            for mm in self.metas
            if mm.get("property", "unknown").startswith("og:")
        }

    def _get_authors(self) -> List[str]:
        """
        _get_authors
        """
        return self.permutive_data.get("authors", [])

    def _get_story(self):
        """
        _get_story
        """
        story = ""
        for paragraph in self.bs.find(class_="RichTextStoryBody RichTextBody").find_all(
            "p"
        ):
            p = paragraph.text
            # replace unicode quotes and quote-like characters with apostrophe
            # https://hexdocs.pm/ex_unicode/Unicode.Category.QuoteMarks.html
            p = re.sub(r"[\u0018-\u0019]", "'", p)
            story += p + "\n"
        return story

    def pickle(self):
        """
        pickle
        """
        with open(self.path(), "wb") as fh:
            pickle.dump(self, fh)

    def upsert(self, client: Any):
        """
        upsert to server via client
        """
        client.upsert_article(self)

    def path(self):
        """path"""
        os.makedirs(f"./daily_news/{self.archive}", exist_ok=True)
        return f"./daily_news/{self.archive}/{self.item_id}.pkl"

    def cache(self):
        """
        cache
            Create a redis cache of the article keyed on URL
            with path to the pickled article
        """
        REDIS.set(self.url, self.path())

    def accept(self, visitor: Any):
        """
        visit
        """
        visitor.visit_article(self)
