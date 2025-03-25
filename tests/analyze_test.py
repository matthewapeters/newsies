"""tests.analyze_test"""

import uuid
import os
import pwd

from newsies.pipelines import analyze_pipeline
from newsies.pipelines.task_status import TASK_STATUS

# pylint: disable=broad-exception-caught


def test__analyze_pipeline():
    """
    test__analyze_pipeline
    """
    task_id = str(uuid.uuid4())
    user_id = pwd.getpwuid(os.getuid())[0]
    TASK_STATUS[task_id] = {
        "session_id": "N/A",
        "status": "queued",
        "task": "analyze",
        "username": user_id,
    }
    try:
        analyze_pipeline(task_id=task_id)
    except Exception as e:
        assert False, f"ERROR: {e}"
