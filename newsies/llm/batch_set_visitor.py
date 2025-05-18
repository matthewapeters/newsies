"""
newsies.visitor.visitors
"""

from abc import abstractmethod
from typing import Any

from newsies.visitor import Visitor

from .batch_set import BatchSet


class BatchSetVisitor(Visitor):
    """BatchSetVisitor"""

    def visit_target(self, target: Any):
        t: BatchSet = target
        self.visit_batch_set(t)

    @abstractmethod
    def visit_batch_set(self, batch_set: Any):
        """
        Visit the BatchSet class and train a machine learning model.
        """
        raise NotImplementedError("visit_batch_set not implemented")
