"""
newsies-analyzer
Content analysis and summarization service
"""

__version__ = "0.2.0"

# Re-export commonly used items
# Note: Import only what exists in this package
try:
    from .pipelines.analyze import analyze_pipeline
except ImportError:
    analyze_pipeline = None

try:
    from .classify import *
except ImportError:
    pass

__all__ = [
    "pipelines",
    "classify",
    "analyze_pipeline",
]
