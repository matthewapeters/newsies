import inflect

from .main import ChromaDBClient

p = inflect.engine()
client = ChromaDBClient()
client.collection = "ordinal_reference"
count = client.collection.count()
# Define range of ordinal numbers to store (can be extended)
ordinal_range = 500
ordinal_references = []
if count < ordinal_range:
    from sentence_transformers import SentenceTransformer

    # Load Sentence Transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Store ordinal references in ChromaDB
    for i in range(1, ordinal_range + 1):
        ordinal_word = p.number_to_words(i, ordinal=True)  # "first", "second", etc.
        ordinal_numeral = p.ordinal(i)  # "1st", "2nd", etc.

        # Embed both forms
        embedding = model.encode(ordinal_word).tolist()

        # Store in ChromaDB
        client.collection.add(
            ids=[f"ordinal_{i}"],  # Unique ID
            embeddings=[embedding],
            metadatas=[{"number": i, "text": ordinal_word, "numeral": ordinal_numeral}],
        )

        ordinal_references.append((ordinal_word, ordinal_numeral, i))


def find_ordinal(text):
    """Find the closest ordinal reference using ChromaDB."""
    query_embedding = model.encode(text).tolist()

    # Perform a similarity search
    results = client.collection.query(
        query_embeddings=[query_embedding], n_results=1  # Get the top match
    )

    if results["documents"] and results["metadatas"][0]:
        best_match = results["metadatas"][0][0]
        return (
            best_match["number"],
            best_match["text"],
            best_match["numeral"],
        )  # Extract data

    return None  # No match found
