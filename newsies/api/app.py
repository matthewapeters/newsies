"""
newsies.api.app

"""

from typing import Dict
import gc
import uuid
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks

from newsies.pipelines import TASK_STATUS, LOCK
from newsies.ap_news.sections import SECTIONS
from newsies.chroma_client import CRMADB
from newsies.targets import HEADLINE

# pylint: disable=import-outside-toplevel
app = FastAPI()


def serve_api():
    """
    serve_api
    """


def run_get_news(task_id: str):
    """
    run_get_news
    """
    from newsies.pipelines import get_news_pipeline

    LOCK.acquire()
    try:
        get_news_pipeline(task_id)
        gc.collect()
    finally:
        LOCK.release()


def run_analyze(task_id: str):
    """
    run_analyze
    """
    from newsies.pipelines import analyze_pipeline

    LOCK.acquire()
    try:
        analyze_pipeline(task_id)
        gc.collect()
    finally:
        LOCK.release()


@app.get("/run/get-news")
async def run_get_news_pipeline(background_tasks: BackgroundTasks):
    """
    run_get_news_pipeline
    run the get-news pipeline
    the pipeline checks the Associated Press website for any articles in each of its sections.
    Articles are then downloaded to local cache and embedded in search engine
    """
    task_id = str(uuid.uuid4())
    TASK_STATUS[task_id] = "queued"
    background_tasks.add_task(run_get_news, task_id)
    return {"message": "getting latest news from Associated Press", "task_id": task_id}


@app.get("/run/analyze")
async def run_analyze_pipeline(background_tasks: BackgroundTasks):
    """
    run_analyze_pipeline
    run the analyze pipeline.
    The pipeline summarizes all stories, searches and adds named enttities and n-grams
    to the search engine
    """
    task_id = str(uuid.uuid4())
    TASK_STATUS[task_id] = "queued"
    background_tasks.add_task(run_analyze, task_id)
    return {
        "message": "analyzing latest news from Associated Press",
        "task_id": task_id,
    }


@app.get("/task-status/{task_id}")
def get_task_status(task_id: str):
    """
    get_task_status
    retrieve the current status of the requested task id
    """
    status = TASK_STATUS.get(task_id, f"task id {task_id} not found")
    return {"task_id": task_id, "status": status}


@app.get("/tasks")
def list_tasks():
    """
    list_tasks
    provides the current set of admin tasks and their status
    """
    return {"newsies_tasks": TASK_STATUS.sorted(), "as_of": datetime.now().isoformat()}


@app.get("/sections")
def list_sections():
    """
    list_sections
    provides a list of sections from the Associated Press website
    """
    return {"news_sections": SECTIONS}


@app.get("/headlines/{section}")
def list_headlines(section: str):
    """
    list_headlines
    returns todays headlines from the requested section
    """
    resp = CRMADB.collection.get(
        where={
            "$and": [
                {"target": {"$eq": HEADLINE}},
                {
                    "$or": [
                        {"section0": {"$eq": section}},
                        {"section1": {"$eq": section}},
                        {"section2": {"$eq": section}},
                    ]
                },
            ]
        }
    )
    headlines: Dict[str] = list(
        {h.strip().replace("‘", "'").replace("’", "'") for h in resp["documents"]}
    )
    headlines = sorted(headlines)

    return {
        "section": section,
        "headlines": headlines,
    }
