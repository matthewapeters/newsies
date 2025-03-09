"""
newsies.chromadb_client.headlines
"""

from typing import Dict, List

from newsies import targets

from .main import ChromaDBClient


def get_all_headlines(collection: str, section: str) -> List[str]:
    """
    get_all_headlines
    """
    client = ChromaDBClient()
    client.collection_name = collection
    resp = client.collection.get(
        where={
            "$and": [
                {"target": {"$eq": targets.HEADLINE}},
                {
                    "$or": [
                        {"section0": {"$eq": section}},
                        {"section1": {"$eq": section}},
                        {"section2": {"$eq": section}},
                    ]
                },
            ]
        }
    )
    headlines: Dict[str] = list(
        {h.strip().replace("‘", "'").replace("’", "'") for h in resp["documents"]}
    )
    return sorted(headlines)
