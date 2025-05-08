"""newsies.utils.fix_article_formats"""

import os
import pickle

from newsies.ap_news.article import Article
from newsies.ap_news.article_format_visitor import ArticleFormatVisitor

# pylint: disable=too-few-public-methods,protected-access,no-member


def fix_article_formats():
    """fix_article_formats"""
    # get all the articles in the daily_news directory
    daily_news_dir = "./daily_news/apnews.com/"
    articles = [
        os.path.join(daily_news_dir, f)
        for f in os.listdir(daily_news_dir)
        if os.path.isfile(os.path.join(daily_news_dir, f))
        and len(f) == 36  # 36 is the length of the item_id string + .pkl
    ]
    print(f"Found {len(articles)} articles to format")
    # clear previous format pipeline entries
    visitor = ArticleFormatVisitor()
    for article_path in articles:
        article: Article = pickle.load(open(article_path, "rb"))
        article.pipelines.pop(ArticleFormatVisitor.step_name)

        visitor.visit(article)
        article.pickle()
        print(f"Formatted {article_path}")


if __name__ == "__main__":
    fix_article_formats()
