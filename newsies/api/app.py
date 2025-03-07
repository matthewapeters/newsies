"""
newsies.api.app

"""

from typing import Callable, Dict
import gc
import uuid
from datetime import datetime
from functools import wraps

from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from fastapi.responses import RedirectResponse
from newsies.redis_client import cache_session, get_session
from newsies.pipelines import TASK_STATUS, LOCK
from newsies.ap_news.sections import SECTIONS
from newsies.chroma_client import CRMADB
from newsies.chromadb_client import ChromaDBClient, collections
from newsies.targets import HEADLINE
from newsies.session.init_session import init_session

# pylint: disable=import-outside-toplevel, broad-exception-caught, unused-argument
app = FastAPI()

SESSION_COOKIE_NAME = "sessionid"
USER_COOKIE_NAME = "usrnm"


def get_session_id(request: Request):
    """Retrieve session ID from cookie or query parameters."""
    return request.cookies.get(SESSION_COOKIE_NAME)


def require_session(endpoint: Callable):
    """Decorator to enforce session requirement."""

    @wraps(endpoint)
    async def wrapper(request: Request, *args, **kwargs):
        sessionid = get_session_id(request)
        if not sessionid:
            login_url = f"/login?redirect={request.url}"
            return RedirectResponse(url=login_url)

        return await endpoint(request, *args, **kwargs)

    return wrapper


def run_get_news(*args, **kwargs):
    """
    run_get_news
    """
    from newsies.pipelines import get_news_pipeline

    LOCK.acquire()
    try:
        get_news_pipeline(*args, **kwargs)
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
async def run_get_news_pipeline(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    run_get_news_pipeline
    run the get-news pipeline
    the pipeline checks the Associated Press website for any articles in each of its sections.
    Articles are then downloaded to local cache and embedded in search engine
    """
    task_id = str(uuid.uuid4())
    username = request.cookies[USER_COOKIE_NAME]
    sess = request.cookies[SESSION_COOKIE_NAME]
    TASK_STATUS[(task_id, sess, username)] = "queued"
    background_tasks.add_task(run_get_news, task_id)
    return {"message": "getting latest news from Associated Press", "task_id": task_id}


@app.get("/run/analyze")
@require_session
async def run_analyze_pipeline(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    run_analyze_pipeline
    run the analyze pipeline.
    The pipeline summarizes all stories, searches and adds named enttities and n-grams
    to the search engine
    """
    task_id = str(uuid.uuid4())
    username = request.cookies[USER_COOKIE_NAME]
    sess = request.cookies[SESSION_COOKIE_NAME]
    TASK_STATUS[(task_id, sess, username)] = "queued"
    background_tasks.add_task(run_analyze, task_id)
    return {
        "message": "analyzing latest news from Associated Press",
        "task_id": task_id,
    }


@app.get("/task-status/{task_id}")
@require_session
async def get_task_status(
    request: Request,
    task_id: str,
):
    """
    get_task_status
    retrieve the current status of the requested task id
    """
    status = TASK_STATUS.get(task_id, f"task id {task_id} not found")
    return {"task_id": task_id, "status": status}


@app.get("/tasks")
@require_session
async def list_tasks(request: Request):
    """
    list_tasks
    provides the current set of admin tasks and their status
    """
    return {"newsies_tasks": TASK_STATUS.sorted(), "as_of": datetime.now().isoformat()}


@app.get("/sections")
@require_session
async def list_sections(request: Request):
    """
    list_sections
    provides a list of sections from the Associated Press website
    """
    return {"news_sections": SECTIONS}


@app.get("/headlines/{section}")
@require_session
async def list_headlines(
    request: Request,
    section: str,
):
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
async def get_index(request: Request):
    """
    get_index
        redirects to /docs
    """
    response = RedirectResponse(url="/docs")
    return response


@app.get("/login/{username}")
@app.get("/login")
async def login(username: str = "", redirect: str = "/"):
    """
    assign a session ID
    """
    if not username.strip():  # Check for empty or whitespace username
        raise HTTPException(
            status_code=403, detail="Username is required. IE: /login/<username>"
        )

    session, _ = init_session(username=username)
    cache_session(session)
    response = RedirectResponse(url=redirect)
    response.set_cookie(key=SESSION_COOKIE_NAME, value=session.id, httponly=True)
    response.set_cookie(key=USER_COOKIE_NAME, value=username, httponly=True)
    return response


@app.get("/collections")
@require_session
async def list_collections(request: Request):
    """
    list_collections
    """
    return {"collections": collections(ChromaDBClient())}


@app.get("/collection/{collection}")
@require_session
async def enable_collection(request: Request, collection: str):
    """enable_collection"""
    try:
        sessid = request.cookies[SESSION_COOKIE_NAME]
        sess = get_session(sessid)
        sess.collection = f"ap_news_{collection}"
        cache_session(sess)
        return {"status": "ok", "message": f"using collection {sess.collection}"}
    except Exception as e:
        return HTTPException(status_code=500, detail=f"ERROR: {e}")
