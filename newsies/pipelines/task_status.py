"""
newsies.pipelines.task_status
"""

from datetime import datetime
import threading
from typing import List, Dict, Union


class AppStatus(dict):
    """
    AppStatus
    """

    def __setitem__(self, key: str, value: Union[str, Dict[str, str]]):
        """
        __setitem__
        :param key: str task id uuid
        :param value: Union[str,Dict[str,str]]
            task details when queued (Dict[str,str]) or status update (str)
        """

        # when the status is a str, retrieve the task status record
        # an update the status - then make the value the task status
        # record
        if isinstance(value, str):
            temp_value = self[key]
            temp_value["status"] = value
            value = temp_value

        # set the timestamp in the task status record
        now = datetime.now().isoformat()
        value["timestamp"] = now

        # write the status to the logfile
        with open("newsies.log", "a", encoding="utf8") as log:
            log.write(f"INFO:\t{now}\tTASK STATUS\t{key}\t{value}")

        super().__setitem__(key, value)  # store the status record

    def sorted(self) -> List[Dict]:
        """
        sort
        render the status in descending order of timestamp
        """
        if len(self) == 0:
            return []
        tasks = [{k: v} for k, v in self.items()]
        tasks.sort(key=lambda t: list(t.values())[0]["timestamp"], reverse=True)
        return tasks


# Dictionary to track task statuses
TASK_STATUS = AppStatus()
LOCK = threading.Lock()
