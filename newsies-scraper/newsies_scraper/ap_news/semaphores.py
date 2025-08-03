"""newsies.ap_news.semaphores"""

import threading
from typing import Callable
from functools import wraps
from multiprocessing import Lock as MpLock


_MTX: threading.Lock = threading.Lock()
_GET_LOCK: threading.Lock = threading.Lock()
_MP_LOCK = MpLock()


def protected_factory(mtx: threading.Lock = _MTX) -> Callable:
    """
    protected wrapper
        Wraps functions with MTX lock
        param: mtx: Lock -- which mutex to use.  Defaults to _MTX
    """

    def wrapper_factory(c: Callable):
        """wrapper_factory"""

        @wraps(c)
        def protected_callable(*args, **kwargs):
            """protected_callable"""
            with mtx:
                return c(*args, **kwargs)

        return protected_callable

    return wrapper_factory


protected = protected_factory()
get_protected = protected_factory(_GET_LOCK)
multi_processing_protected = protected_factory(_MP_LOCK)
