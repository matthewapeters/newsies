"""
newsies.pipelines
"""

from typing import Dict

from .analyze import *
from .get_news import *

# Dictionary to track task statuses
TASK_STATUS: Dict[str, str] = {}
