"""
newsies.ap_news.embedding_visitor
"""

from typing import List

import torch
from sentence_transformers import SentenceTransformer

from newsies.ap_news.article import Article

# pylint: disable=broad-exception-caught

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2", device=DEVICE_STR
)  # Fast and good quality


class EmbeddingVisitor:
    """
    EmbeddingVisitor
    """

    def __init__(self):
        self.visitor = None

    def visit(self, node: any):
        """
        visit
        """
        node.accept(self)

    def visit_article(self, article: Article):
        """
        visit_article
        """
        to_embed = list(article.section_headlines.values())
        to_embed.append(article.story)
        to_embed.append(article.summary)
        to_embed.extend(article.named_entities)
        embeddings: List[float] = []
        for te in to_embed:
            embeddings.append(embedding_model.encode(te))
        article.embeddings = embeddings
