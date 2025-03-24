"""
newsies.ap_news.document_structures
"""

from typing import List

# pylint: disable=too-few-public-methods, unused-argument


class Headline:
    """
    Headline
    """

    url: str = None
    headline: str = None
    section: str = None

    def __init__(self, *args, **kwargs):
        self.url = kwargs["url"]
        self.headline = kwargs["headline"]
        self.section = kwargs["section"]

    def dump(self) -> dict:
        return {
            "url": self.url,
            "headline": self.headline,
            "section": self.section,
        }


class Document:
    """
    Document
    """

    url: str = None
    #    uri: str = None
    date: str = None
    source: str = None
    text: str = None
    target: str = None
    headlines: List[str] = None
    sections: List[str] = None

    def __init__(self, *args, **kwargs):
        # the following are created in the initial routines
        self.url = kwargs["url"]
        # self.uri = kwargs["uri"]
        self.date = kwargs["date"]
        self.target = kwargs["target"]
        self.headlines = kwargs["headlines"]
        self.sections = kwargs["sections"]
        # when targets == SUMMARY, added at summarization
        #     (summaries are small enough not to persist)
        self.text = kwargs.get("text", "")

    def dump(self) -> dict:
        """
        dump
        """
        return {
            "url": self.url,
            # "uri": self.uri,
            "date": self.date,
            "target": self.target,
            "headlines": self.headlines.copy(),
            "sections": self.sections.copy(),
            "text": self.text,
        }
