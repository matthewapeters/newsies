import pytest

from newsies.chromadb_client import find_ordinal

test_data = [
    ("pick the second one", 2),
    ("read the fifth article", 5),
    ("get me the first three hundred articles", 300),
]


@pytest.mark.parametrize("input, expected", test_data)
def test__find_ordinal(input, expected):

    ordinals = find_ordinal(input)
    assert (
        ordinals[0] == expected
    ), f"expected {expected} from {input}, but got {ordinals[0]}"
