import sys
from datetime import datetime


def usage():
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
                from newsies.ap_news import (
                    get_latest_news,
                    news_loader,
                    news_summarizer,
                )

                headlines = get_latest_news()
                news_loader(headlines)
                news_summarizer(headlines)
            case "serve":
                pass
            case "cli":
                from newsies.cli import read_news

                date_archive = datetime.now().strftime(r"%Y-%m-%d")
                if len(sys.argv) > 3:
                    date_archive = sys.argv[2]
                read_news(date_archive)
            case _:
                usage()
