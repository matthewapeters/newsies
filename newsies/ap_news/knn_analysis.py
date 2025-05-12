"""
newsies.ap_news.knn_analysis
"""

from newsies.ap_news.clusters import Cluster, get_cluster

# pylint: disable=broad-exception-caught, broad-exception-raised


def generate_knn_graph():
    """
    generate_knn_graph
    """
    try:
        cluster: Cluster = get_cluster()
    except Exception as e:
        raise Exception(f"archive.get_archive: ERROR: {e}") from e
    try:
        cluster.refresh()
    except Exception as e:
        raise Exception(f"archive.refresh: ERROR: {e}") from e
    try:
        cluster.build_knn()
    except Exception as e:
        raise Exception(f"archive.build_knn: ERROR: {e}") from e
    try:
        cluster.dump()  # backup the cluster
    except Exception as e:
        raise Exception(f"archive.dump: ERROR: {e}") from e
