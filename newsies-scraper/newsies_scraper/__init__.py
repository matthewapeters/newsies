"""
newsies-scraper
News scraping and article ingestion service
"""

__version__ = "0.2.0"

# Re-export commonly used items
from .ap_news import *
from .pipelines import get_articles_pipeline

__all__ = [
    "ap_news",
    "pipelines",
    "get_articles_pipeline",
]
