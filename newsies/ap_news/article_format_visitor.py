"""newsies.llm.article_format_visitor"""

from datetime import datetime

from newsies.ap_news.article import Article
from newsies.visitor.visitor import BatchSetVisitor

from ..llm.specs import (
    ARTICLE_END,
    ARTICLE_START,
    ITEM_ID_END,
    ITEM_ID_START,
    SECTION_END,
    SECTION_START,
    TITLE_END,
    TITLE_START,
    PUBDATE_END,
    PUBDATE_START,
    AUTHOR_END,
    AUTHOR_START,
)


class ArticleFormatVisitor(BatchSetVisitor):
    """Visitor to format an article for the LLM."""

    def __init__(
        self,
    ) -> None:
        super().__init__(
            target_type=Article,
            history_path="./daily_news/apnews.com/formatted_articles.pkl",
            step_name="format_article",
        )
        self.article = None

    def visit(self, target: Article) -> str:
        """Visit an article and return its formatted string."""
        match target:
            case Article():
                self.visit_article(target)
            case _:
                raise TypeError(f"Expected Article, got {type(target)}")

    def visit_article(self, article: Article) -> str:
        """Visit an article and return its formatted string."""
        if self.step_name in article.pipelines:
            return
        section_titles = [
            (
                f"{SECTION_START}{section or 'front page'}{SECTION_END}: "
                f"{TITLE_START}{headline}{TITLE_END}"
            )
            for section, headline in article.section_headlines.items()
        ]
        if len(section_titles) > 0:
            section_titles = "\n".join(section_titles)
        else:
            section_titles = ""

        authors = [f"{AUTHOR_START}{author}{AUTHOR_END}" for author in article.authors]
        if len(authors) > 0:
            authors = "\n".join(authors)
        else:
            authors = ""
        article.formatted = (
            f"{ITEM_ID_START}{article.item_id}{ITEM_ID_END}\n"
            f"{PUBDATE_START}{article.publish_date.isoformat()}{PUBDATE_END}\n"
            f"{section_titles}\n"
            f"{authors}\n"
            f"{ARTICLE_START}{article.story}{ARTICLE_END}\n"
        )

        article.pipelines[self.step_name] = datetime.now().isoformat()

    def visit_batch_set(self, batch_set):
        """
        visit_batch_set
        this cannot be reached, but is required by the visitor pattern
        to satisfy the abstract method
        """
        return

    @property
    def step_name(self) -> str:
        """
        step_name property
        """
        return self._step_name
