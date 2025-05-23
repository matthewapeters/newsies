"""
tests.archives_test
"""

from threading import Lock

from newsies.ap_news.archive import Archive, protected_factory, get_archive

# pylint: disable=broad-exception-caught


def test__articles():
    """test__articles"""
    archive = Archive()
    archive.refresh()
    assert len(archive.collection) > 0
    archive.build_knn()
    # DO NOT DUMP THE ARCHIVE AS PART OF THE TEST!


def test__protected_factory():
    """test__protected"""
    my_mtx = Lock()

    p = protected_factory(my_mtx)

    @p
    def test(*args, **kwargs):
        assert my_mtx.locked(), "expected my_mtx to be locked"
        return len(args), len(kwargs)

    t = test(1, **{"p1": "this", "p2": "that"})
    assert t[0] == 1
    assert t[1] == 2


def test__archive_to_dict():
    """test__archive_to_dict"""
    arch: Archive = get_archive()
    try:
        j = arch.to_dict()
        assert j != "{}"
    except Exception as e:
        assert False, f"ERROR: {e}"


def test__archive_build_batches():
    """test__archive_build_batches"""
    arch: Archive = get_archive()
    try:
        batches = arch.build_batches()
        assert len(batches) > 0
    except Exception as e:
        assert False, f"ERROR: {e}"
