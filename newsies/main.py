"""
newsies.main
"""

import sys
from typing import Dict

from newsies.document_structures import Document


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
                print("\nGET NEWS\n")
                print("\n\t- retrieving headlines\n")
                from newsies.ap_news.latest_news import get_latest_news

                headlines: Dict[str, Document] = get_latest_news()

                print("\n\t- news loader\n")
                from newsies.ap_news.latest_news import news_loader

                news_loader(headlines)

                print("\n\t- headlines loader\n")
                from newsies.ap_news.latest_news import headline_loader

                headline_loader(headlines)
            case "analyze":
                print("\nANALYZE NEWS\n")
                print("\t- retrieving headlines\n")
                from newsies.ap_news.latest_news import get_latest_news

                headlines: Dict[str, Document] = get_latest_news()

                print("\n\t- summarizing stories")
                from newsies.ap_news import batch_news_summarizer

                batch_news_summarizer(headlines)

                print("\n\t- detecting ngrams specific to news sections\n")
                from newsies.ap_news import analyze_ngrams_per_section

                analyze_ngrams_per_section(headlines)

                print("\n\t- computing tf-idf for ngrams\n")
                from newsies.ap_news import compute_tfidf

                compute_tfidf()
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
