"""
newsies.ap_news.knn_analysis
"""

from newsies.ap_news.archive import get_archive

# pylint: disable=broad-exception-caught, broad-exception-raised


def generate_knn_graph():
    """
    generate_knn_graph
    """
    try:
        archive = get_archive()
    except Exception as e:
        raise Exception(f"archive.get_archive: ERROR: {e}") from e
    try:
        archive.refresh()
    except Exception as e:
        raise Exception(f"archive.refresh: ERROR: {e}") from e
    try:
        archive.build_knn()
    except Exception as e:
        raise Exception(f"archive.build_knn: ERROR: {e}") from e
    try:
        archive.dump()  # backup the archive
    except Exception as e:
        raise Exception(f"archive.dump: ERROR: {e}") from e
