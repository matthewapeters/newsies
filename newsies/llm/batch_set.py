"""
nwesies.llm.batch_set
This module contains the BatchSet class, which is used to manage batches of
data for training and testing machine learning models. The BatchSet class provides
a datastructure for storing batches of data, against which several visitors can operate
for the purpose of training and testing machine learning models.
"""

from typing import Any, Dict, List, Tuple, Union
import pickle

from datasets import Dataset


class BatchSet:
    """
    BatchSet class is used to manage batches of data for training and testing machine
    learning models.
    It provides a datastructure for storing batches of data, against which several
    visitors can operate for the purpose of training and testing machine learning models.
    """

    path: str = "./train_test/batch_set.pkl"

    def __init__(self, batches: List[Tuple[str]]):
        """
        Initialize the BatchSet class.
        """
        self.batches: List[Tuple[str]] = batches
        self.embeddings: List[List[float]] = []
        self.metadatas: List[List[Dict[str, Union[str, int, float]]]] = []
        self.data_sets: List[Dataset] = []

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
        Return the number of batches in the BatchSet.
        """
        return len(self.batches)

    def __getitem__(self, index):
        """
        Get a batch by index.
        """
        return self.batches[index]

    def __iter__(self):
        """
        Iterate over the batches in the BatchSet.
        """
        return iter(self.batches)

    def accept(self, visitor: Any):
        """
        Accept a visitor to operate on the BatchSet.
        """
        return visitor.visit_batch_set(self)
