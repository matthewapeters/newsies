"""
newsies.api.app

"""

import json
import gc
import uuid
from datetime import datetime
import threading
from pydantic import BaseModel


from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi import FastAPI, BackgroundTasks, Request, HTTPException, Path, APIRouter
from fastapi.responses import RedirectResponse
import uvicorn


from newsies.session import Session, get_session_params
from newsies.session.init_session import init_session
from newsies.pipelines import TASK_STATUS
from newsies.ap_news.sections import SECTIONS
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
        get_articles_pipeline(*args, **kwargs)
        gc.collect()
    finally:
        RUN_LOCK.release()


def run_analyze(task_id: str, archive: str = None):
    """
    run_analyze
    """
    from newsies.pipelines import analyze_pipeline

    RUN_LOCK.acquire()
    try:
        analyze_pipeline(task_id, archive)
        gc.collect()

    finally:
        RUN_LOCK.release()


def run_train_model(task_id):
    """
    run_train_model
    """
    from newsies.pipelines.train_model import train_model_pipeline

    RUN_LOCK.acquire()
    try:
        train_model_pipeline(task_id)
        gc.collect()
    finally:
        RUN_LOCK.release()


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
    background_tasks.add_task(run_get_articles, task_id)
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


@router_v1.get("/run/train-llm")
@require_session
async def run_train_llm(request: Request, background_tasks: BackgroundTasks):
    """
    run_train_llm
    Train the model in the background
    Will work backwards through the training data
    """
    task_id = str(uuid.uuid4())
    username = request.cookies[USER_COOKIE_NAME]
    sess = request.cookies[SESSION_COOKIE_NAME]
    TASK_STATUS[task_id] = {
        "session_id": sess,
        "status": "queued",
        "task": "train-llm",
        "username": username,
    }
    background_tasks.add_task(run_train_model, task_id)
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


@router_v1.get("/sections")
@require_session
async def list_sections(request: Request):
    """
    list_sections
    provides a list of sections from the Associated Press website
    """
    return {"news_sections": SECTIONS}


# @router_v1.get("/headlines/{section}")
# @require_session
# async def list_headlines(
#     request: Request,
#     section: str,
# ):
#     """
#     list_headlines
#     returns headlines from the requested section
#     """
#     sessid = request.cookies[SESSION_COOKIE_NAME]
#     session: Session = Session(**get_session_params(sessid))
#     qa = {
#         "context": "NEW",
#         "target": HEADLINE,
#         "section": section,
#         "quantity": "ALL",
#     }
#     output = session.query(query_analysis=qa)
#     cache_session(session)
#     return output


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
