"""newsies.llm.tokenizer"""

from typing import List

from transformers import AutoTokenizer

from .specs import (
    _BASE_MODEL_NAME,
)

# pylint: disable=invalid-name, global-statement

TOKENIZER: AutoTokenizer = None


def tokenize(prompt: str) -> List:
    """
    Tokenizes the input string using the global tokenizer.
    """
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(_BASE_MODEL_NAME)
    return TOKENIZER(prompt, return_tensors="pt")["input_ids"]


def decode(tokens: List) -> str:
    """
    Decodes the input tokens using the global tokenizer.
    """
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(_BASE_MODEL_NAME)
    return TOKENIZER.decode(tokens[0], skip_special_tokens=True)
