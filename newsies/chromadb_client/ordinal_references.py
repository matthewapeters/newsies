"""
newsies.chromadb_client.ordinal_references
"""

import inflect
from sentence_transformers import SentenceTransformer

from .main import ChromaDBClient

# pylint: disable=invalid-name

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

p = inflect.engine()
client = ChromaDBClient()
client.collection_name = "ordinal_reference"
# Define range of ordinal numbers to store (can be extended)
ORDINAL_LIMIT = 1000


def load_ordinals():
    """
    load_ordinals
     - rune once if these are not already loaded
    """

    def embed(ordinal_text):
        # Embed both forms
        embedding = model.encode(ordinal_text).tolist()

        # Store in ChromaDB
        client.collection.add(
            ids=[f"ordinal_text_{i}"],  # Unique ID
            embeddings=[embedding],
            documents=[ordinal_text],
            metadatas=[{"number": i, "text": ordinal_text}],
        )

    # Store ordinal references in ChromaDB
    for i in range(ORDINAL_LIMIT):
        embed(make_text_ordinal(i))
        embed(make_numeric_ordinal(i))


def find_ordinal(text):
    """Find the closest ordinal reference using ChromaDB."""
    query_embedding = model.encode(text).tolist()

    # Perform a similarity search
    results = client.collection.query(
        query_embeddings=[query_embedding], n_results=2  # Get the top match
    )

    if results["documents"] and results["metadatas"][0]:
        best_match = results["metadatas"][0][0]
        return {
            "number": best_match["number"],
            "text": best_match["text"],
            "distance": results["distances"][0][0],
        }
    return None  # No match found


oridnal_suffix = {
    0: "th",
    1: "st",
    2: "nd",
    3: "rd",
    4: "th",
    5: "th",
    6: "th",
    7: "th",
    8: "th",
    9: "th",
}


named_ordinals = {
    0: "zeroth",
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eigteenth",
    19: "nineteenth",
    20: "twentieth",
    30: "thirtieth",
    40: "fortieth",
    50: "fiftieth",
    60: "sixtieth",
    70: "seventieth",
    80: "eightieth",
    90: "ninetieth",
}

multiples_of_tens = {
    2: "twenty",
    3: "thirty",
    4: "forty",
    5: "fifty",
    6: "sixty",
    7: "seventy",
    8: "eighty",
    9: "ninety",
}
integers = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


def make_numeric_ordinal(nbr: int) -> str:
    suffix: str = oridnal_suffix[nbr % 10]
    return f"{nbr}{suffix}"


def make_text_ordinal(nbr: int) -> str:
    """
    convert ordinas from index 0 to 999 into english text
    """
    if nbr in named_ordinals:
        return named_ordinals[nbr]
    hundreds_factor = int((nbr % 1000) / 100)
    tens_factor = int((nbr % 100) / 10)
    remaining = (nbr % 100) - (tens_factor * 10)
    if nbr > 99:
        hundred_ord = "hundred" if tens_factor > 0 or remaining > 0 else "hundredth"
        hundreds = f"{integers[hundreds_factor]} {hundred_ord}"
        if tens_factor == 0 and remaining == 0:
            return f"{hundreds}"
        if nbr - (hundreds_factor * 100) in named_ordinals:
            i = nbr - (hundreds_factor * 100)
            return f"{hundreds} {named_ordinals[i]}"
        return (
            f"{hundreds} {multiples_of_tens[tens_factor]}-{named_ordinals[remaining]}"
        )
    return f"{multiples_of_tens[tens_factor]}-{named_ordinals[remaining]}"


_count = client.collection.count()
if _count + 1 < ORDINAL_LIMIT:
    load_ordinals()
