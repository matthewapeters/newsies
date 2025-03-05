"""
newsies.api.app

"""

import gc
import uuid
import asyncio

from fastapi import FastAPI, BackgroundTasks

from pipelines import TASK_STATUS

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

    get_news_pipeline(task_id)
    gc.collect()


def run_analyze(task_id: str):
    """
    run_analyze
    """
    from newsies.pipelines import analyze_pipeline

    analyze_pipeline(task_id)
    gc.collect()


@app.get("/run/get-news")
def run_get_news_pipeline(background_tasks: BackgroundTasks):
    """
    run_get_news_pipeline
    """
    task_id = str(uuid.uuid4())
    background_tasks.add_task(asyncio.create_task, run_get_news(task_id))
    return {"message": "getting latest news from Associated Press", "task_id": task_id}


@app.get("/run/analyze")
def run_analyze_pipeline(background_tasks: BackgroundTasks):
    """
    run_analyze_pipeline
    """
    task_id = uuid.uuid4()
    background_tasks.add_task(asyncio.create_task, run_analyze(task_id))
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
