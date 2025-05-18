"""newsies.ap_news.archive_visitor"""

from abc import abstractmethod
from typing import Any

from newsies.visitor import Visitor

from .archive import Archive


class ArchiveVisitor(Visitor):
    """ArchiveVisitor"""

    def visit_target(self, target: Any):
        t: Archive = target
        self.visit_archive(t)

    @abstractmethod
    def visit_archive(self, archive: Archive):
        """
        visit_archive
            Visit the Archive class instance and process articles
        """
