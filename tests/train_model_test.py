"""
tests.train_model_test
"""

from datetime import datetime
import os


from newsies.pipelines.train_model import (
    generate_qa_pairs,
    train_model,
    test_lora,
    get_train_and_test_data,
)


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
    test_data_dir = f"./test_data/{datetime.now().strftime(r'%Y%m%d%H%M')}"
    os.makedirs(test_data_dir, exist_ok=True)
    test_data.to_parquet(f"{test_data_dir}/test_data.parquet")

    assert lora_dir.startswith("./lora_adapters")
    assert len(test_data) > 0


def test__test_lora():
    """
    test__test_lora
    """
    # TODO we should get the last entry in lora_adapters.txt
    lora_dir = "./lora_adapaters/mistal_lora_202503180906/"
    # TODO we should get the most recent test_data
    generate_qa_pairs(batch_size=11, number_of_questions=3)
    _, test_data = get_train_and_test_data()

    # run the test against the model/adapter
    results = test_lora(lora_dir, test_data)
    assert len(results) > 0, "expected results to be larger than 0"
    assert results["success"], f"expected success, got {results}"
