"""newsies.utils.fix_test_data.py
Fix test data for newsies
"""

import os
from shutil import rmtree

from newsies.ap_news.archive import get_archive
from newsies.llm import DatasetFormatter
from newsies.llm import BatchSet


def fix_test_data():
    """fix_test_data"""
    base_dir = "./train_test"
    rmtree(base_dir, ignore_errors=True)
    state_file = "./training_data/formatted_dates.pkl"
    if os.path.exists(state_file):
        os.remove(state_file)

    batches = get_archive().build_batches()
    batch_set = BatchSet(batches)

    # Split each publish date into train, test, token_train, and token_test
    # datasets.  These are saved to disk as parquet files.
    # The files are named with the publish date and the type of dataset.
    v = DatasetFormatter()
    v.visit(batch_set)
    assert os.path.exists(base_dir), f"expected {base_dir} to exist"

    pub_date_folders = os.listdir(base_dir)
    pub_dates = list(batches.keys())
    assert len(pub_date_folders) == len(pub_dates)


if __name__ == "__main__":
    fix_test_data()
