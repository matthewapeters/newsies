"""
tests.train_model_test
"""

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
    lora_dir, test_data = train_model()
    assert lora_dir.startswith("./lora_adapters")
    assert len(test_data) > 0


def test__test_lora():
    """
    test__test_lora
    """
    lora_dir = "./lora_adapaters/mistal_lora_202503180906/"
    generate_qa_pairs()
    _, test_data = get_train_and_test_data()
    results = test_lora(lora_dir, test_data)
    assert len(results) > 0, "expected results to be larger than 0"
    assert results["success"], f"expected success, got {results}"
