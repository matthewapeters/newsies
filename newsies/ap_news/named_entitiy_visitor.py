"""
newsies.ap_news.named_entity_visitor
"""

from collections import Counter

import spacy
from spacy.tokens import Doc
from sentence_transformers import SentenceTransformer
import torch

from newsies.ap_news.article import Article

#  Entity Type	Description
#
#  CARDINAL	    Numerals that don’t fall under another type
#  DATE	        Absolute or relative dates, time expressions
#  EVENT	    Named events, like hurricanes, wars, sports events
#  FAC	        Buildings, airports, highways, bridges, etc.
#  GPE	        Countries, cities, states
#  LANGUAGE	    Named human languages
#  LAW	        Named legal documents, laws, treaties
#  LOC	        Non-GPE locations, such as mountains, water bodies
#  MONEY	    Monetary values, including currency symbols
#  NORP	        Nationalities, religious groups, or political groups
#  ORDINAL	    First, second, third, etc.
#  ORG	        Companies, agencies, institutions, etc.
#  PERCENT	    Percentage expressions, e.g., "50%"
#  PERSON	    People, including fictional characters
#  PRODUCT	    Objects, vehicles, foods, etc. (not services)
#  QUANTITY	    Measurements, including units (e.g., "5kg")
#  TIME	        Specific times of day
#  WORK_OF_ART	Books, songs, films, paintings, etc.


# pylint: disable=broad-exception-caught, broad-exception-raised

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2", device=DEVICE_STR
)  # Fast and good quality


# Load spaCy model for English
nlp = spacy.load("en_core_web_sm")

DATE = "DATE"
EVENT = "EVENT"
FAC = "FAC"
GPE = "GPE"
LANGUAGE = "LANGUAGE"
LAW = "LAW"
LOC = "LOC  "
MONEY = "MONEY"
NORP = "NORP"
ORG = "ORG"
PERCENT = "PRECENT"
PERSON = "PERSON"
PRODUCT = "PRODUCT"
QUANTITY = "QUANTITY"
TIME = "TIME"
WORK_OF_ART = "WORK_OF_ART"

SUPPORTED_ENTITY_TYPES = {
    DATE,
    EVENT,
    FAC,
    GPE,
    LANGUAGE,
    LAW,
    LOC,
    MONEY,
    NORP,
    ORG,
    PERCENT,
    PERSON,
    PRODUCT,
    QUANTITY,
    TIME,
    WORK_OF_ART,
}


class NamedEntityVisistor:
    """NamedEntityVisistor"""

    def __init__(self):
        pass

    def visit(self, node):
        """
        visit
        """
        node.accept(self)

    def visit_article(self, article: Article):
        """
        visit_article
        """
        docs = [h for hl in list(article.section_headlines.values()) for h in hl]
        docs.append(article.story)
        docs.append(article.summary)
        d = ""
        try:
            for d in docs:
                article.named_entities.extend(detect_named_entities(nlp(d)))
        except Exception as e:
            raise Exception(f"NamedEntityVisistor {e}  {d}") from e


def detect_named_entities(doc: Doc) -> Counter:
    """
    detect_named_entities
    """

    named_entities = Counter()
    for ent in doc.ents:
        if any(token.pos_ in SUPPORTED_ENTITY_TYPES for token in ent):
            named_entities[ent.text] += 1  # Count named entities
        else:
            # Remove stopwords from non-proper-noun entities
            tokens = [token.text for token in ent if not token.is_stop]
            if tokens:
                named_entities[" ".join(tokens)] += 1

    return named_entities
