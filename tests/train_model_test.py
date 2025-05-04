"""
tests.train_model_test
"""

import os
from typing import Dict
from shutil import rmtree

from datasets import Dataset
import datasets
import pandas as pd

from newsies.ap_news.archive import Archive, get_archive

from newsies.llm.train_model import (
    # generate_qa_pairs,
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
from newsies.llm import (
    BatchSet,
    BatchRetriever,
    DataFramer,
    QuestionGenerator,
    load_qa_from_parquet,
    DatasetFormatter,
    ModelTrainer,
)
from newsies.lora_adapter import get_latest_lora_adapter

# pylint: disable=broad-exception-caught


def test__build_batches():
    """
    test__build_batches
    """
    arch = get_archive()
    assert isinstance(arch, Archive)
    batches = arch.build_batches()
    assert isinstance(batches, Dict)
    assert len(batches) > 0, f"expected batches to be larger than 0, got {batches}"
    sample = list(batches.values())[0]
    assert isinstance(
        sample, list
    ), f"expected sample batch to be a set, got {type(sample)}"
    batch_set = BatchSet(batches)
    assert isinstance(batch_set, BatchSet)
    br = BatchRetriever()
    assert isinstance(br, BatchRetriever)
    br.visit(batch_set)
    #    assert len(batch_set.embeddings) == len(
    #        batch_set.batches
    #    ), f"expected {len(batch_set.batches)}, got {len(batch_set.embeddings)}"
    assert len(batch_set.metadatas) == len(
        batch_set.batches
    ), f"expected {len(batch_set.batches)}, got {len(batch_set.metadatas)}"

    df = DataFramer()
    assert isinstance(df, DataFramer)
    df.visit(batch_set)
    assert len(batch_set.data_sets) == len(batch_set), (
        f"expected len(batch_set.data_sets) == {len(batch_set)}, "
        f"got {len(batch_set.data_sets)}"
    )
    assert isinstance(
        batch_set.data_sets[0], Dataset
    ), f"expected batch_set.data_sets[0] to be a Dataset, got {batch_set.data_sets[0]}"
    assert (
        len(batch_set.data_sets[0]) > 0
    ), f"expected batch_set.data_sets[0] to be larger than 0, got {batch_set.data_sets[0]}"
    assert len(batch_set.data_sets[0].features) > 0, (
        f"expected batch_set.data_sets[0].features to be larger than 0, "
        f"got {batch_set.data_sets[0].features}"
    )
    assert isinstance(batch_set.data_sets[0].features, datasets.Features), (
        f"expected batch_set.data_sets[0].features to be a Features, "
        f"got {batch_set.data_sets[0].features}"
    )
    assert isinstance(batch_set.data_sets[0].features["doc"], datasets.Value), (
        f"expected batch_set.data_sets[0].features['doc'] to be a Value, "
        f"got {batch_set.data_sets[0].features['doc']}"
    )
    v = QuestionGenerator()
    v.visit(batch_set)
    base_dir = "./training_data"
    pub_dates = [d for d in os.listdir(base_dir) if os.path.isdir(f"{base_dir}/{d}")]
    assert (
        len(pub_dates) > 0
    ), "expected at least one publish date folder under ./training_data"
    pub_dates.sort()
    most_recent = pub_dates[-1]
    batches = os.listdir(f"./training_data/{most_recent}/")
    batches.sort()
    assert len(batches) > 0, "expected at least one batch"

    sample = load_qa_from_parquet(
        file_path=f"./training_data/{most_recent}/{batches[-1]}/"
    )

    assert len(batch_set.batches[most_recent]) == len(
        batches
    ), f"expected a folder for each batch for {most_recent}"

    # assert (
    #    batch_set.data_sets[0]["doc"][0] == sample[0]["doc"]
    # ), "expected parquet doc to match dataset"

    assert (
        len(sample) % v.number_of_questions == 0 and len(sample) > 0
    ), f"assert, sample should be greater than 0 and multiple of {v.number_of_questions}"

    assert len(
        [s["question"] for s in sample if s.get("question", "N/A") != "N/A"]
    ) == len(sample), "assert each entry has a question"

    # training data should be organized on disk by publish date and then by
    # batch, allowing training to prioritize most recent and most populated
    # clusters
    # It is currently saved under training_data/{batch_id}
    # change to training_data/{publish_date}/{batch_id}  -- batch_id is ordered by
    # cluster size in descending order, so largest clusters (over history) get
    # priority in training

    v = DatasetFormatter()
    assert isinstance(v, DatasetFormatter)
    v.visit(batch_set)

    # Train the model
    v = ModelTrainer()
    # setting these allows status update as we iterate through the batches
    # if either is not set, the status will not update
    v.visit(batch_set)


def test__dataset_formatter():
    """
    test__dataset_formatter
    """
    base_dir = "./train_test"

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


def test__trainer_visitor():
    """
    test__trainer_visitor
    """
    batches = get_archive().build_batches()
    batch_set = BatchSet(batches)

    # Train the model
    v = ModelTrainer()
    # setting these allows status update as we iterate through the batches
    # if either is not set, the status will not update
    v.visit(batch_set)


def test__generate_qa_pairs():
    """
    test__generate_qa_pairs
        Uses Flat T5 to generate questions regarding articles and named entities.
        These questions and the expected answers from the context are used to
        fine-tune the model's LoRA adapter.
    """
    # generate_qa_pairs(batch_size=11, number_of_questions=3)


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
