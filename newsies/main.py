import sys


def usage():
    print(
        """

                Newsies
          Interactive News Explorer

    USAGE:

        newsies [routine] [options]

        ROUTINES:
          * get-news - daily routine to gather and analyze news
          * cli - interactive command-line agent
          * serve - serve the newsies API for integration



          """
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
    else:
        match sys.argv[1]:
            case "get-news":
                from newsies.news_loader.main import (
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
                from newsies.cli.main import read_news

                read_news()
            case _:
                usage()
