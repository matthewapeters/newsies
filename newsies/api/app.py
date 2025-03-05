"""
newsies.api.app

"""

from fastapi import FastAPI
from newsies.pipelines import get_news_pipeline, analyze_pipeline


app = FastAPI()


def serve_api():
    """
    serve_api
    """


@app.get("/run/get-news")
def run_get_news_pipeline():
    """
    run_get_news_pipeline
    """
    get_news_pipeline()


@app.get("/run/analyze")
def run_analyze_pipeline():
    """
    run_analyze_pipeline
    """
    analyze_pipeline()
