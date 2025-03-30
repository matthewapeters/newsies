"""
tests.articles_test
"""

from newsies.archive import Archive


def test__articles():
    """test__articles"""
    archive = Archive()
    archive.refresh()
    assert len(archive.collection) > 0
    archive.build_knn()
