"""
newsies.api.app

"""

from typing import Callable, Dict
import gc
import uuid
from datetime import datetime
from functools import wraps

from fastapi import FastAPI, BackgroundTasks, Request, HTTPException, Path, APIRouter
from fastapi.responses import RedirectResponse
from newsies.redis_client import cache_session, get_session
from newsies.pipelines import TASK_STATUS, LOCK
from newsies.ap_news.sections import SECTIONS
from newsies.chromadb_client import ChromaDBClient, collections
from newsies.targets import HEADLINE
from newsies.session import Session, Turn
from newsies.session.init_session import init_session

# pylint: disable=import-outside-toplevel, broad-exception-caught, unused-argument, protected-access
app = FastAPI()

router_v1 = APIRouter()

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


def run_analyze(task_id: str, archive: str = None):
    """
    run_analyze
    """
    from newsies.pipelines import analyze_pipeline

    LOCK.acquire()
    try:
        analyze_pipeline(task_id, archive)
        gc.collect()
    finally:
        LOCK.release()


@router_v1.get("/run/get-news")
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
    TASK_STATUS[task_id] = {
        "session_id": sess,
        "status": "queued",
        "task": "get-news",
        "username": username,
    }
    background_tasks.add_task(run_get_news, task_id)
    return {"message": "getting latest news from Associated Press", "task_id": task_id}


@router_v1.get("/run/analyze")
@require_session
async def run_analyze_pipeline_today(
    request: Request, background_tasks: BackgroundTasks
):
    """run_analyze_pipeline_today"""
    return await run_analyze_pipeline(
        request, background_tasks, datetime.now().strftime(r"%Y-%m-%d")
    )


@router_v1.get("/run/analyze/{archive}")
@require_session
async def run_analyze_pipeline(
    request: Request,
    background_tasks: BackgroundTasks,
    archive: str = Path(..., regex=r"\d{4}-\d{2}-\d{2}"),  # Enforces YYYY-MM-DD format
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
    if archive is None:
        archive = datetime.now().strftime(r"%Y-%m-%d")
    TASK_STATUS[task_id] = {
        "archive": archive,
        "session_id": sess,
        "status": "queued",
        "task": "analyze",
        "username": username,
    }
    background_tasks.add_task(run_analyze, task_id, archive)
    return {
        "message": "analyzing latest news from Associated Press",
        "task_id": task_id,
    }


@router_v1.get("/task-status/{task_id}")
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


@router_v1.get("/tasks")
@require_session
async def list_tasks(request: Request):
    """
    list_tasks
    provides the current set of admin tasks and their status
    """
    return {"newsies_tasks": TASK_STATUS.sorted(), "as_of": datetime.now().isoformat()}


@router_v1.get("/sections")
@require_session
async def list_sections(request: Request):
    """
    list_sections
    provides a list of sections from the Associated Press website
    """
    return {"news_sections": SECTIONS}


@router_v1.get("/headlines/{section}")
@require_session
async def list_headlines(
    request: Request,
    section: str,
):
    """
    list_headlines
    returns headlines from the requested section
    """
    sessid = request.cookies[SESSION_COOKIE_NAME]
    session: Session = get_session(sessid)
    client = ChromaDBClient()
    client.collection_name = (
        session.collection or f"ap_news_{datetime.now().strftime(r'%Y-%m-%d')}"
    )

    resp = client.collection.get(
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

    output = {
        "section": section,
        "headlines": headlines,
    }
    turn = Turn()
    turn._paged_document_map = [{"0": output}]
    session.add_history(turn)
    cache_session(session)

    return output


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
async def login(username: str = "", redirect: str = "/"):
    """
    login and create a session for your user
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


@router_v1.get("/collections")
@require_session
async def list_collections(request: Request):
    """
    list_collections
        lists the archived collections in the system
    """
    return {"collections": collections(ChromaDBClient())}


@router_v1.get("/collection/{collection}")
@require_session
async def enable_collection(request: Request, collection: str):
    """
    enable_collection
        selects a collection date for your session
    """
    try:
        sessid = request.cookies[SESSION_COOKIE_NAME]
        sess = get_session(sessid)
        sess.collection = f"ap_news_{collection}"
        cache_session(sess)
        return {"status": "ok", "message": f"using collection {sess.collection}"}
    except Exception as e:
        return HTTPException(status_code=500, detail=f"ERROR: {e}")


app.include_router(router_v1, prefix="/v1")
