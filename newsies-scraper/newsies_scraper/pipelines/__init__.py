"""
newsies-scraper pipelines
News scraping pipeline implementation
"""

from .get_articles import get_articles_pipeline

__all__ = [
    "get_articles_pipeline",
]
