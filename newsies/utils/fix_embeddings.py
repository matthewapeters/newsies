"""newsies.utils.fix_embeddings"""

import pickle

from newsies.ap_news.article import Article
from newsies.ap_news.embedding_visitor import EmbeddingVisitor
from newsies.ap_news.index_visitor import IndexVisitor

# pylint: disable=unused-argument, broad-exception-caught


def fix_embeddings():
    """
    fix_embeddings
    """
    with open(
        "./daily_news/apnews.com/missing_embeddings.txt", "r", encoding="utf8"
    ) as missing_fh:
        articles = [a.strip() for a in missing_fh.readlines() if len(a.strip()) == 32]
    print(f"Found {len(articles)} articles with missing embeddings")
    if len(articles) == 0:
        print("No articles to fix")
        return
    try:
        for article_id in articles:
            with open(f"./daily_news/apnews.com/{article_id.strip()}.pkl", "rb") as fh:
                article: Article = pickle.load(fh)
            if not isinstance(article, Article):
                print(f"Error: {article_id.strip()} is not an Article")
                continue
            v = EmbeddingVisitor()
            v.visit(article)
            article.pickle()
            iv = IndexVisitor()
            iv.collection_name = "apnews.com"
            iv.visit(article)
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
