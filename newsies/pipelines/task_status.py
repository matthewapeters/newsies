"""
newsies.pipelines.task_status
"""

from datetime import datetime
import threading
from typing import List, Dict


class AppStatus(dict):
    """
    AppStatus
    """

    def __setitem__(self, key, value):
        """
        __setitem__
        """
        now = datetime.now().isoformat()
        username: str = None
        sessionid: str = None
        if isinstance(key, tuple):
            sessionid = key[1]
            username = key[2]
            key = key[0]
        else:
            sessionid = self[key]["session_id"]
            username = self[key]["user_name"]

        enhanced_value = {
            "status": value,
            "timestamp": now,
            "session_id": sessionid,
            "user_name": username,
        }
        with open("newsies.log", "a", encoding="utf8") as log:
            log.write(f"INFO:\t{now}\tTASK STATUS\t{key}\t{value}")
        super().__setitem__(key, enhanced_value)

    def sorted(self) -> List[Dict]:
        """
        sort
        render the status in descending order of timestamp
        """
        if len(self) == 0:
            return []
        tasks = [{k: v} for k, v in self.items()]
        tasks.sort(key=lambda t: list(t.values())[0]["timestamp"])
        return tasks


# Dictionary to track task statuses
TASK_STATUS = AppStatus()
LOCK = threading.Lock()
