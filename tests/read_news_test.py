"""tests.read_news_test"""


def test__read_news():
    """
    test__read_news
    """
    from newsies.cli.main import cli_read_news

    # Test the function
    try:
        cli_read_news("what is the latest on Donald Trump and Elon Musk?")
    except SystemExit as e:
        assert e.code == 0, f"Expected exit code 0, but got {e.code}"
