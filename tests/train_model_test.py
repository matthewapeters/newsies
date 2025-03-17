"""
tests.train_model_test
"""

from newsies.pipelines.train_model import generate_qa_pairs, train_model


def test__generate_qa_pairs():
    """
    test__generate_qa_pairs2
    """
    generate_qa_pairs(batch_size=11, number_of_questions=3)


def test__train_model():
    """
    test__train_model
    """
    train_model()
