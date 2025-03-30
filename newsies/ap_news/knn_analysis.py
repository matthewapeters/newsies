"""
newsies.ap_news.knn_analysis
"""

from newsies.ap_news.archive import Archive


def generate_knn_graph():
    """
    generate_knn_graph
    """
    archive = Archive()
    archive.refresh()
    archive.build_knn()
