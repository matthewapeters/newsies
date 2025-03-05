"""
newsies.main
"""

import sys

# pylint: disable=import-outside-toplevel


def usage():
    """usage"""
    print(
        """

                Newsies
          Interactive News Explorer

    USAGE:

        newsies [routine] [options]

        ROUTINES:
          * get-news - daily routine to gather and analyze news
          * cli <archive date> - interactive command-line agent
            - archive date in the form YYYY-MM-DD
            - defaults to today
          * serve - serve the newsies API for integration



          """
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
    else:
        match sys.argv[1]:
            case "get-news":
                from .pipelines import get_news_pipeline

                get_news_pipeline()
            case "analyze":
                from .pipelines import analyze_pipeline

                analyze_pipeline()
            case "serve":
                from .api import serve_api

                serve_api()
            case "cli":
                from newsies.cli import read_news

                date_archive: str = None
                if len(sys.argv) > 3:
                    date_archive = sys.argv[2]
                read_news(date_archive)
            case _:
                usage()
