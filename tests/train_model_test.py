"""
tests.train_model_test
"""

import os
import pwd
import uuid
from newsies.pipelines.task_status import TASK_STATUS
from newsies.pipelines.train_model import train_model_pipeline

# pylint: disable=broad-exception-caught


def test__train_model_pipeline():
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
        train_model_pipeline(task_id=task_id)
    except Exception as e:
        assert False, f"ERROR: {e}"
