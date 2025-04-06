"""
tests.train_model_test
"""

import os
from typing import Dict

import pandas as pd
import datasets
from datasets import Dataset

from newsies.llm.train_model import (
    generate_qa_pairs,
    train_model,
    test_lora,
    get_train_and_test_data,
    load_training_data,
    format_dataset,
    get_latest_training_data,
    TEST,
    TTEST,
    TRAIN_DATA_TYPES,
)
from newsies.lora_adapter import get_latest_lora_adapter

# pylint: disable=broad-exception-caught


def test__generate_qa_pairs():
    """
    test__generate_qa_pairs
        Uses Flat T5 to generate questions regarding articles and named entities.
        These questions and the expected answers from the context are used to
        fine-tune the model's LoRA adapter.
    """
    generate_qa_pairs(batch_size=11, number_of_questions=3)


def test__get_train_and_test_data():
    """
    test__get_train_and_test_data
    """
    split_dataset = get_train_and_test_data()
    assert len(split_dataset) == 4, f"expected 4 datasets, got {len(split_dataset)}"
    for t in TRAIN_DATA_TYPES:
        assert t in split_dataset, f"expected {t} in split_dataset, got {split_dataset}"
    base = "./train_test"
    dirs = os.listdir(base)
    assert len(dirs) > 0, "expected dirs to be larger than 0"
    dirs.sort()
    latest_dir = dirs[-1]
    dirs = os.listdir(f"{base}/{latest_dir}")
    assert len(dirs) == 4, f"expected 4 dirs, got {len(dirs)}"
    for t in TRAIN_DATA_TYPES:
        assert t in dirs, f"expected {t} in dirs, got {dirs}"
        assert os.path.exists(
            f"{base}/{latest_dir}/{t}/data.parquet"
        ), f"expected {t}.csv to exist"


def test__train_model():
    """
    test__train_model

        {'train_runtime': 83919.1656, 'train_samples_per_second': 6.434,
        'train_steps_per_second': 6.434, 'train_loss': 0.17590146832617198,
        'epoch': 3.0}

        100%|██████████| 539934/539934 [23:18:39<00:00,  6.43it/s]
    """
    # train the model and save/log the lora adapter
    lora_dir, test_data = train_model()
    assert lora_dir.startswith("lora_adapters")
    assert len(test_data) > 0
    latest_lora_dir = get_latest_lora_adapter()
    assert latest_lora_dir == lora_dir, f"expected {latest_lora_dir} == {lora_dir}"


def test__test_latest_lora_adapter():
    """
    test__test_latest_lora_adapter
    """
    lora_dir = get_latest_lora_adapter()
    try:
        data_dict: Dict[str, Dataset] = get_latest_training_data([TEST, TTEST])
    except OSError:
        get_train_and_test_data()
        data_dict = get_latest_training_data([TEST, TTEST])

    # run the test against the model/adapter
    results = test_lora(lora_dir, data_dict)
    assert len(results) > 0, "expected results to be larger than 0"
    assert results["success"], f"expected success, got {results}"


def test__load_training_data():
    """
    test__load_training_data
    """
    df = load_training_data()
    assert df is not None
    assert len(df) > 0
    assert len(df.columns) > 0
    keys = df.keys()
    assert isinstance(keys, pd.Index)
    for c in ["question", "answer", "doc", "uri", "section", "headline"]:
        assert c in keys, f"expected column {c} in dataframe, got {keys}"


def test__format_dataset():
    """
    test__format_data
    """
    df = load_training_data()
    try:
        ds_dict: datasets.DatasetDict = format_dataset(df)
        assert len(ds_dict) == 4, f"expected 4 datasets, got {len(ds_dict)}"
    except Exception as e:
        assert False, f"failed to format dataset: {e}"
