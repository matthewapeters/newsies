"""
newsies.api.app

"""

import json
import gc
import uuid
from datetime import datetime
import threading
from typing import Callable
from pydantic import BaseModel


from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi import FastAPI, BackgroundTasks, Request, HTTPException, Path, APIRouter
from fastapi.responses import RedirectResponse
import uvicorn


from newsies.session import Session, get_session_params
from newsies.session.init_session import init_session
from newsies.pipelines import TASK_STATUS
from newsies.ap_news.archive import get_archive, Archive


from newsies.api.session import (
    SESSION_COOKIE_NAME,
    USER_COOKIE_NAME,
    require_session,
)
from newsies.api.dashboard import (
    DASHBOARD_APP,
    get_knn_graph_data,
    get_knn_graph,
    get_most_recent_graph,
)

# pylint: disable=import-outside-toplevel, broad-exception-caught, unused-argument, protected-access


RUN_LOCK = threading.Lock()

app = FastAPI()
router_v1 = APIRouter()
app.mount("/dashboard/", WSGIMiddleware(DASHBOARD_APP.server))
app.mount(
    "/_dash-component-suites/", require_session(WSGIMiddleware(DASHBOARD_APP.server))
)


def run_get_articles(*args, **kwargs):
    """
    run_get_articles
    """
    from newsies.pipelines import get_articles_pipeline

    RUN_LOCK.acquire()
    try:
        if "parent_task_id" in kwargs:
            parent_task_id = kwargs["parent_task_id"]
            if parent_task_id and parent_task_id in TASK_STATUS:
                TASK_STATUS[parent_task_id] = "running get_articles_pipeline"
        get_articles_pipeline(task_id=kwargs["task_id"])
        gc.collect()
    except Exception as e:
        if "parent_task_id" in kwargs:
            TASK_STATUS[kwargs["parent_task_id"]] = f"exception: {e}"
            parent_task_id = kwargs["parent_task_id"]
            if parent_task_id and parent_task_id in TASK_STATUS:
                TASK_STATUS[parent_task_id] = "get_articles_pipeline failed"
        else:
            print(f"get_articles_pipeline failed: {e}")
    finally:
        RUN_LOCK.release()


def run_analyze(task_id: str, archive: str = None, parent_task_id: str = None):
    """
    run_analyze
    """
    from newsies.pipelines import analyze_pipeline

    RUN_LOCK.acquire()
    try:
        if parent_task_id and parent_task_id in TASK_STATUS:
            last_status: str = TASK_STATUS[parent_task_id]["status"]
            if last_status.endswith("failed"):
                return
            TASK_STATUS[parent_task_id] = "running analyze_pipeline"
        analyze_pipeline(task_id, archive)
        gc.collect()
    except Exception as e:
        if parent_task_id and parent_task_id in TASK_STATUS:
            TASK_STATUS[parent_task_id] = f"exception: {e}"
            TASK_STATUS[parent_task_id] = "analyze_pipeline failed"
        else:
            print(f"analyze_pipeline failed: {e}")
    finally:
        RUN_LOCK.release()


def run_train_model(task_id, parent_task_id: str = None):
    """
    run_train_llm
    """
    from newsies.pipelines.train_model import train_model_pipeline

    RUN_LOCK.acquire()
    try:
        if parent_task_id and parent_task_id in TASK_STATUS:
            last_status: str = TASK_STATUS[parent_task_id]["status"]
            if last_status.endswith("failed"):
                return
            TASK_STATUS[parent_task_id] = "running train_model_pipeline"
        train_model_pipeline(task_id)
        gc.collect()
        TASK_STATUS[task_id]["status"] = "complete"
    except Exception as e:
        if parent_task_id and parent_task_id in TASK_STATUS:
            TASK_STATUS[parent_task_id] = f"exception: {e}"
            TASK_STATUS[parent_task_id] = "failed"
        else:
            print(f"train_model_pipeline failed: {e}")
    finally:
        RUN_LOCK.release()


def queue_task(
    task_name: str,
    username: str,
    sess: str,
    task: Callable,
    background_tasks: BackgroundTasks,
    **kwargs,
) -> str:
    """
    queue_task
    """
    task_id = str(uuid.uuid4())
    TASK_STATUS[task_id] = {
        "session_id": sess,
        "status": "queued",
        "task": task_name,
        "username": username,
    }
    background_tasks.add_task(task, task_id=task_id, **kwargs)

    return task_id


@router_v1.get("/run/daily-pipeline")
@require_session
async def run_daily_pipeline(request: Request, background_tasks: BackgroundTasks):
    """
    run_daily_pipeline
    run the daily pipeline
    * the pipeline checks the Associated Press website for any articles in each of its sections.
    Articles are then downloaded to local cache and embedded in search engine
    * the pipeline summarizes all stories, searches and adds named enttities and n-grams
    to the search engine
    * the pipeline trains the model
    """
    parent_task_id = str(uuid.uuid4())
    username = request.cookies[USER_COOKIE_NAME]
    sess = request.cookies[SESSION_COOKIE_NAME]

    get_news_task_id = queue_task(
        "get-news",
        username,
        sess,
        run_get_articles,
        background_tasks,
        parent_task_id=parent_task_id,
    )

    analyze_task_id = queue_task(
        "analyze",
        username,
        sess,
        run_analyze,
        background_tasks,
        archive=datetime.now().strftime(r"%Y-%m-%d"),
        parent_task_id=parent_task_id,
    )

    train_task_id = queue_task(
        "train-llm",
        username,
        sess,
        run_train_model,
        background_tasks,
        parent_task_id=parent_task_id,
    )

    status = {
        "session_id": sess,
        "status": "queued",
        "task": "daily-pipeline",
        "username": username,
        "tasks": {
            "get_news": get_news_task_id,
            "analyze": analyze_task_id,
            "train": train_task_id,
        },
    }

    TASK_STATUS[parent_task_id] = status
    return status


@router_v1.get("/run/get-news")
@require_session
async def run_get_news_pipeline(
    request: Request,
    background_tasks: BackgroundTasks,
    parent_task_id: str = None,
):
    """
    run_get_news_pipeline
    run the get-news pipeline
    the pipeline checks the Associated Press website for any articles in each of its sections.
    Articles are then downloaded to local cache and embedded in search engine
    """
    username = request.cookies[USER_COOKIE_NAME]
    sess = request.cookies[SESSION_COOKIE_NAME]
    task_id = queue_task(
        "get-news",
        username,
        sess,
        run_get_articles,
        background_tasks,
        parent_task_id=parent_task_id,
    )
    return {"message": "getting latest news from Associated Press", "task_id": task_id}


@router_v1.get("/run/analyze")
@require_session
async def run_analyze_pipeline_today(
    request: Request, background_tasks: BackgroundTasks, parent_task_id: str = None
):
    """run_analyze_pipeline_today"""
    username = request.cookies[USER_COOKIE_NAME]
    sess = request.cookies[SESSION_COOKIE_NAME]

    task_id = queue_task(
        "analyze",
        username,
        sess,
        run_analyze,
        background_tasks,
        archive=datetime.now().strftime(r"%Y-%m-%d"),
        parent_task_id=parent_task_id,
    )
    return {
        "message": "analyzing latest news from Associated Press",
        "task_id": task_id,
    }


@router_v1.get("/run/analyze/{archive}")
@require_session
async def run_analyze_pipeline(
    request: Request,
    background_tasks: BackgroundTasks,
    archive: str = Path(..., regex=r"\d{4}-\d{2}-\d{2}"),  # Enforces YYYY-MM-DD format
    parent_task_id: str = None,
):
    """
    run_analyze_pipeline
    run the analyze pipeline.
    The pipeline summarizes all stories, searches and adds named enttities and n-grams
    to the search engine
    """
    username = request.cookies[USER_COOKIE_NAME]
    sess = request.cookies[SESSION_COOKIE_NAME]
    task_id = queue_task(
        "analyze",
        username,
        sess,
        run_analyze,
        background_tasks,
        archive=archive or datetime.now().strftime(r"%Y-%m-%d"),
        parent_task_id=parent_task_id,
    )
    return {
        "message": "analyzing latest news from Associated Press",
        "task_id": task_id,
    }


@router_v1.get("/run/train-llm")
@require_session
async def run_train_llm(
    request: Request, background_tasks: BackgroundTasks, parent_task_id: str = None
):
    """
    run_train_llm
    Train the model in the background
    Will work backwards through the training data
    """
    username = request.cookies[USER_COOKIE_NAME]
    sess = request.cookies[SESSION_COOKIE_NAME]

    task_id = queue_task(
        "train-llm",
        username,
        sess,
        run_train_model,
        background_tasks,
        parent_task_id=parent_task_id,
    )
    return {"message": "training the LLM on latest news data", "task_id": task_id}


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

    session = init_session(username=username)
    session.cache_session()
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
    arch: Archive = get_archive()
    return {"collection_archive": arch.to_dict()}


@router_v1.get("/collection/{collection}")
@require_session
async def enable_collection(request: Request, collection: str):
    """
    enable_collection
        selects a collection date for your session
    """
    try:
        sessid = request.cookies[SESSION_COOKIE_NAME]
        sess = Session(**get_session_params(sessid))
        sess.collection = f"ap_news_{collection}"
        sess.cache_session()
        return {"status": "ok", "message": f"using collection {sess.collection}"}
    except Exception as e:
        return HTTPException(status_code=500, detail=f"ERROR: {e}")


@router_v1.get("/session")
@require_session
async def dump_session(request: Request):
    """dump_session"""
    session_id = request.cookies[SESSION_COOKIE_NAME]
    s = Session(**get_session_params(session_id))
    return s.dump()


class Prompt(BaseModel):
    """
    Prompt
    """

    prompt: str


@router_v1.post("/prompt")
async def post_prompt(request: Request, user_prompt: Prompt):
    """post_prompt"""
    session_id = request.cookies[SESSION_COOKIE_NAME]
    session = Session(**get_session_params(session_id))
    # response = session.query(user_prompt.prompt)
    session.cache_session()
    return {"ERR": "Not Implemented Yet"}


@router_v1.get("/get-knn-graph")
def get_graph_data():
    """
    Returns graph data in Cytoscape JSON format.
    """
    return json.dumps(get_knn_graph_data(get_data=get_knn_graph))


@router_v1.get("/get-most-recent-knn-graph")
def get_most_recent_knn_graph():
    """
    Returns graph data in Cytoscape JSON format.
    """
    return json.dumps(get_knn_graph_data(get_data=get_most_recent_graph))


app.include_router(router_v1, prefix="/v1")

if __name__ == "__main__":
    uvicorn.run(app)
