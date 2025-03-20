"""
tests.train_model_test
"""

import pandas as pd
import datasets

from newsies.pipelines.train_model import (
    generate_qa_pairs,
    train_model,
    test_lora,
    get_train_and_test_data,
    get_latest_lora_adapter,
    get_latest_test_data,
    load_training_data,
    format_dataset,
)

# pylint: disable=broad-exception-caught


def test__generate_qa_pairs():
    """
    test__generate_qa_pairs2
    """
    generate_qa_pairs(batch_size=11, number_of_questions=3)


def test__train_model():
    """
    test__train_model
    """
    # train the model and save/log the lora adapter
    lora_dir, test_data = train_model()
    assert lora_dir.startswith("./lora_adapters")
    assert len(test_data) > 0


def test__test_latest_lora_adapter():
    """
    test__test_latest_lora_adapter
    """
    lora_dir = get_latest_lora_adapter()
    try:
        test_data: pd.DataFrame = get_latest_test_data()
    except OSError:
        get_train_and_test_data()
        test_data = get_latest_test_data()

    test_data = test_data.head(10)
    # run the test against the model/adapter
    results = test_lora(lora_dir, test_data)
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
