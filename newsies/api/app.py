"""
newsies.api.app

"""

from typing import Callable, Dict
import gc
import uuid
from datetime import datetime
from functools import wraps

from fastapi import FastAPI, BackgroundTasks, Request, Response, Depends
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from newsies.pipelines import TASK_STATUS, LOCK
from newsies.ap_news.sections import SECTIONS
from newsies.chroma_client import CRMADB
from newsies.targets import HEADLINE
from newsies.session.init_session import init_session

# pylint: disable=import-outside-toplevel
app = FastAPI()

SESSION_COOKIE_NAME = "sessionid"


def get_session_id(request: Request):
    """Retrieve session ID from cookie or query parameters."""
    session_id = request.cookies.get(SESSION_COOKIE_NAME) or request.query_params.get(
        SESSION_COOKIE_NAME
    )
    return session_id


def require_session(endpoint: Callable):
    """Decorator to enforce session requirement."""

    @wraps(endpoint)
    async def wrapper(request: Request, *args, **kwargs):
        session_id = get_session_id(request)
        if not session_id:
            login_url = f"/login?redirect={request.url}"
            return RedirectResponse(url=login_url)
        return await endpoint(request, session_id=session_id, *args, **kwargs)

    return wrapper


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
@require_session
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
@require_session
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
@require_session
def get_task_status(task_id: str):
    """
    get_task_status
    retrieve the current status of the requested task id
    """
    status = TASK_STATUS.get(task_id, f"task id {task_id} not found")
    return {"task_id": task_id, "status": status}


@app.get("/tasks")
@require_session
def list_tasks():
    """
    list_tasks
    provides the current set of admin tasks and their status
    """
    return {"newsies_tasks": TASK_STATUS.sorted(), "as_of": datetime.now().isoformat()}


@app.get("/sections")
@require_session
def list_sections(request: Request):
    """
    list_sections
    provides a list of sections from the Associated Press website
    """
    return {"news_sections": SECTIONS}


@app.get("/headlines/{section}")
@require_session
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


@app.get("/")
@require_session
def get_index():
    """
    get_index
        redirects to /docs
    """
    response = RedirectResponse(url="/docs")
    return response


@app.get("/login/{username}")
async def login(username: str, redirect: str = "/"):
    """
    assign a session ID
    """
    session, _ = init_session(username=username)
    response = RedirectResponse(url=f"{redirect}?{SESSION_COOKIE_NAME}={session.id}")
    response.set_cookie(key=SESSION_COOKIE_NAME, value=session.id, httponly=True)
    return response
