"""
newsies-api
FastAPI gateway service and web dashboard
"""

__version__ = "0.2.0"

# Re-export commonly used items
# Note: Import only what exists in this package
try:
    from .api.app import app
except ImportError:
    app = None

try:
    from .api.dashboard import *
except ImportError:
    pass

__all__ = [
    "app",
    "dashboard",
]
