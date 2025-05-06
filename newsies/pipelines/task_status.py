"""
newsies.pipelines.task_status
"""

import json
from datetime import datetime, timedelta

import threading
from typing import List, Dict, Union

# pylint: disable=fixme, consider-using-with

LOCK = threading.Lock()


def protected(c: callable):
    """
    protected
        wraps function calls with mutex LOCK
    """

    def protected_callable(*args, **kwargs):
        """protected"""
        LOCK.acquire()
        c(*args, **kwargs)
        LOCK.release()

    return protected_callable


class AppStatus(dict):
    """
    AppStatus
    """

    @protected
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
        # TODO - add python logging
        with open("newsies.log", "a", encoding="utf8") as log:
            status_record = json.dumps({"task_id": key, **value})
            log.write(f"INFO:\t{status_record}\n")

        super().__setitem__(key, value)  # store the status record

    def sorted(self, complete_retention: timedelta = timedelta(hours=12)) -> List[Dict]:
        """
        sort
        render the status in descending order of timestamp
        """
        if len(self) == 0:
            return []
        threshold = datetime.now() - complete_retention
        for k, v in self.items():
            if v["status"].startswith("error") or v["status"] in ("complete", "failed"):
                # remove tasks older than complete_retention
                if datetime.fromisoformat(v["timestamp"]) <= threshold:
                    del self[k]
        tasks = [{k: v} for k, v in self.items()]
        tasks.sort(key=lambda t: list(t.values())[0]["timestamp"], reverse=True)
        return tasks


# Dictionary to track task statuses
TASK_STATUS = AppStatus()
