"""newsies.lora_adapter"""

import os
from typing import Dict, List
from datasets import Dataset

import pandas as pd

TRAIN = "train"
TEST = "test"
TTRAIN = "token_train"
TTEST = "token_test"
TRAIN_DATA_TYPES = [TRAIN, TEST, TTRAIN, TTEST]


def get_latest_lora_adapter():
    """
    get_latest_lora_adapter
    """
    with open("./lora_adapters.txt", "rb") as file:
        file.seek(-2, os.SEEK_END)

        # If the file is empty, return an empty string
        if file.tell() == 0:
            return ""

        offset = -2
        whence = os.SEEK_END
        while file.read(1) != b"\n":
            try:
                file.seek(offset, whence)
            except (
                OSError
            ):  # Handle cases where the file size is smaller than the offset
                file.seek(0)  # Go to the beginning of the file
                return file.readline().decode("utf8").strip()
            whence = os.SEEK_CUR
        return file.readline().decode("utf-8").strip()


def get_latest_training_data(types: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    get_latest_train_data
    """
    try:
        base_dir = "./train_test/"
        dirs = os.listdir(base_dir)
        if len(dirs) == 0:
            raise OSError(
                "No test data found. Please run get_train_and_test_data() first."
            )
        dirs.sort()
        latest_dir = dirs[-1]
        train_dict: Dict[str, Dataset] = {}
        if types is None or len(types) == 0:
            types = TRAIN_DATA_TYPES
        for d in types:
            train_dict[d] = Dataset.from_pandas(
                pd.read_parquet(
                    f"{base_dir}/{latest_dir}/{d}/data.parquet"
                ).reset_index(drop=True)
            )
    except OSError as e:
        raise OSError(
            "No test data found. Please run get_train_and_test_data() first."
        ) from e
    return train_dict
