"""
tests.train_model_test
"""

from datetime import datetime
from newsies.pipelines.train_model import generate_qa_pairs


def test__generate_qa_pairs():
    """
    test__generate_qa_pairs2
    """
    print(datetime.now())
    generate_qa_pairs()
    print(datetime.now())
