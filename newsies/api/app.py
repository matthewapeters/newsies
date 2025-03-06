"""
newsies.api.app

"""

import gc
import uuid
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks

from newsies.pipelines import TASK_STATUS, LOCK

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
    """
    task_id = str(uuid.uuid4())
    TASK_STATUS[task_id] = "queued"
    background_tasks.add_task(run_get_news, task_id)
    return {"message": "getting latest news from Associated Press", "task_id": task_id}


@app.get("/run/analyze")
async def run_analyze_pipeline(background_tasks: BackgroundTasks):
    """
    run_analyze_pipeline
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
    """
    status = TASK_STATUS.get(task_id, f"task id {task_id} not found")
    return {"task_id": task_id, "status": status}


@app.get("/tasks")
def list_tasks():
    """
    list_tasks
    """
    return {"newsies_tasks": TASK_STATUS.sorted(), "as_of": datetime.now().isoformat()}
