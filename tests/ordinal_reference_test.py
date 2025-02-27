import pytest

from newsies.chromadb_client import find_ordinal

test_data = [
    ("pick the second one", 2),
    ("read the fifth article", 5),
    ("get me the three-hundred and first article", 301),
]


@pytest.mark.parametrize("input, expected", test_data)
def test__find_ordinal(input, expected):
    ordinals = find_ordinal(input)
    assert ordinals is not None, "expected to find a result, got None"
    assert (
        expected == ordinals["number"]
    ), f"expected {expected}, got {ordinals['number']}"
