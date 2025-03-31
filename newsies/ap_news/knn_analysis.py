"""
newsies.ap_news.knn_analysis
"""

from newsies.ap_news.archive import get_archive


def generate_knn_graph():
    """
    generate_knn_graph
    """
    archive = get_archive()
    archive.refresh()
    archive.build_knn()
    archive.dump()  # backup the archive
