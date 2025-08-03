"""
newsies.visitor.visitor
"""

from typing import Any, Dict, Tuple
import os
import pickle
from abc import ABC, abstractmethod

# These imports will be provided by the packages that use this visitor
# from ..ap_news.archive import Archive
# from ..llm.batch_set import BatchSet


class Visitor(ABC):
    """
    Visitor class is used to visit different types of objects.
    This is a base class for all visitors.
    """

    def __init__(self, target_type: type, history_path: str, step_name: str):
        # key is publish date, value is a list of batches
        # that have been trained
        self.history: Dict[int, Tuple[str]]
        self._history_path: str = history_path
        self._target_class: type = target_type
        self._step_name: str = step_name

        if os.path.exists(self._history_path):
            try:
                with open(self._history_path, "rb") as f:
                    self.history = pickle.load(f)
            except EOFError:
                # if the file is empty, create a new history
                self.history = {}
                self.dump_history()
        else:
            if not os.path.exists(os.path.dirname(self._history_path)):
                os.makedirs(os.path.dirname(self._history_path), exist_ok=True)
            self.history = {}
            self.dump_history()

        self._status: Dict = None
        self._task_id: str = None

    def dump_history(self):
        """
        Dump the trained dates to a file."""
        with open(self._history_path, "wb") as f:
            pickle.dump(self.history, f)

    @property
    def status(self) -> Dict:
        """
        status property
        """
        return self._status

    @status.setter
    def status(self, status: Dict):
        """
        status property
        """
        self._status = status

    @property
    def task_id(self) -> str:
        """
        task_id property
        """
        return self._task_id

    @task_id.setter
    def task_id(self, task_id: str):
        """
        task_id property
        """
        self._task_id = task_id

    def update_status(
        self,
        status: str = "",
    ):
        """
        Update the status of the task.
        """
        if self._status is not None and self._task_id is not None:
            self._status[self._task_id] = (
                f"running - step: {self._step_name}: status: {status}"
            )

    def visit(self, target: Any):
        """
        Visit the BatchSet class and train a machine learning model.
        """
        match target:
            case self._target_class():
                self.visit_target(target)
            case _:
                raise TypeError(
                    f"ModelTrainer only accepts {self._target_class}, got {type(target)}"
                )

    @abstractmethod
    def visit_target(self, target: Any):
        """visit_target"""
        raise NotImplementedError("visit_batch_set not implemented")

    @property
    def step_name(self) -> str:
        """step_name"""
        return self._step_name
