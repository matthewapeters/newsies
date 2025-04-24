"""
nwesies.llm.batch_set
This module contains the BatchSet class, which is used to manage batches of
data for training and testing machine learning models. The BatchSet class provides
a datastructure for storing batches of data, against which several visitors can operate
for the purpose of training and testing machine learning models.
"""

from typing import Any, Dict, Iterable, List, Tuple, Union, Set
import pickle

from datasets import Dataset

from newsies.ap_news.article import Article


class BatchSet:
    """
    BatchSet class is used to manage batches of data for training and testing machine
    learning models.
    It provides a datastructure for storing batches of data, against which several
    visitors can operate for the purpose of training and testing machine learning models.
    """

    path: str = "./train_test/batch_set.pkl"

    def __init__(self, batches: Dict[int, Tuple[str]]):
        """
        Initialize the BatchSet class.
        """
        self.batches: Dict[int, List[Tuple[str]]] = batches
        # self.embeddings: List[List[float]] = []
        self.metadatas: List[List[Dict[str, Union[str, int, float]]]] = []
        self.data_sets: List[Dataset] = []
        self.articles: List[Article] = []

    def save(self):
        """
        Save the BatchSet to a file.
        """
        with open(self.path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load() -> "BatchSet":
        """
        Load the BatchSet from a file.
        """
        with open(BatchSet.path, "rb") as f:
            return pickle.load(f)

    def __len__(self):
        """
        Return the number of datasets in the BatchSet.
        """
        return len(self.data_sets)

    def __getitem__(self, index):
        """
        Get a batch by index.
        """
        return self.batches[index]

    def __iter__(self) -> Iterable[Tuple[Set[str], Dict[str, Any], List[Article]]]:
        """
        Iterate over the batches in the BatchSet.
        """
        return iter(zip(self.batches.items(), self.metadatas, self.articles))

    def accept(self, visitor: Any):
        """
        Accept a visitor to operate on the BatchSet.
        """
        return visitor.visit_batch_set(self)
