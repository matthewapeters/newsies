"""
newsies.pipelines.analyze
"""


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
