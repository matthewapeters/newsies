"""
tests.archives_test
"""

from threading import Lock

from newsies.ap_news.archive import Archive, protected


def test__articles():
    """test__articles"""
    archive = Archive()
    archive.refresh()
    assert len(archive.collection) > 0
    archive.build_knn()


def test__protected():
    """test__protected"""
    my_mtx = Lock()

    @protected(my_mtx)
    def test(*args, **kwargs):
        return len(args, len(kwargs))

    t = test(1, **{"p1": "this", "p2": "that"})
    assert t[0] == 1
    assert t[1] == 1
