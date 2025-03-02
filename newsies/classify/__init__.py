"""
newsies.classify
"""

from .classification_heuristics import (
    ACTION_HEURISTICS,
    TARGET_HEURISTICS,
    ONE_MANY_ALL_HEURISTICS,
    NEWS_SECTION_HEURISTICS,
    QUERY_OR_REFERENCE_HEURISTICS,
)
from .main import prompt_analysis, categorize_text
