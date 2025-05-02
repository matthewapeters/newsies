"""newsies.utils.fix_embeddings"""

import pickle

from newsies.ap_news.article import Article
from newsies.ap_news.embedding_visitor import EmbeddingVisitor

# pylint: disable=unused-argument, broad-exception-caught


def fix_embeddings():
    """
    fix_embeddings
    """
    with open(
        "./daily_news/apnews.com/missing_embeddings.txt", "r", encoding="utf8"
    ) as missing_fh:
        articles = missing_fh.readlines()
    print(f"Found {len(articles)} articles with missing embeddings")
    try:
        for article_id in articles:
            with open(f"./daily_news/apnews.com/{article_id.strip()}.pkl", "rb") as fh:
                article: Article = pickle.load(fh)
            v = EmbeddingVisitor()
            v.visit(article)
            article.pickle()
            print(f"Fixed {article_id.strip()}")
        with open(
            "./daily_news/apnews.com/missing_embeddings.txt", "w", encoding="utf8"
        ) as missing_fh:
            missing_fh.write("")
        print("All articles fixed")
    except Exception as err:
        print(f"Error: {err}")


if __name__ == "__main__":
    fix_embeddings()
