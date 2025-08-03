"""newsies.api.session"""

from typing import Callable
from functools import wraps

from requests import Request
from fastapi.responses import RedirectResponse


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
