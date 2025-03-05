"""
newsies.main
"""

import sys
import gc


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


def analyze_pipeline():
    """
    analyze_pipeline
    """
    print("\nANALYZE NEWS\n")
    # print("\t- retrieving headlines\n")
    # from newsies.ap_news.latest_news import get_latest_news
    # headlines: Dict[str, Document] = get_latest_news()

    from newsies.ap_news import (
        batch_news_summarizer,
        analyze_ngrams_per_section,
        compute_tfidf,
    )

    print("\n\t- summarizing stories")
    batch_news_summarizer()

    print("\n\t- detecting ngrams specific to news sections\n")
    analyze_ngrams_per_section()

    print("\n\t- computing tf-idf for ngrams\n")
    compute_tfidf()


def get_news_pipeline():
    """
    get_news_pipeline
    """
    from newsies.ap_news.latest_news import (
        get_latest_news,
        news_loader,
        headline_loader,
    )

    print("\nGET NEWS\n")
    print("\n\t- retrieving headlines\n")
    # get the latest news links from AP press and pickle for downstream use
    get_latest_news()

    print("\n\t- news loader\n")
    news_loader()

    print("\n\t- headlines loader\n")
    headline_loader()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
    else:
        match sys.argv[1]:
            case "get-news":
                get_news_pipeline()
            case "analyze":
                analyze_pipeline()
            case "serve":
                pass
            case "cli":
                from newsies.cli import read_news

                date_archive: str = None
                if len(sys.argv) > 3:
                    date_archive = sys.argv[2]
                read_news(date_archive)
            case _:
                usage()
