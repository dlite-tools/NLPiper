"""Tokenizer Module."""

from typing import List

__all__ = ["BasicTokenizer"]


class Tokenizer:
    """Abstract class to Tokenizers."""

    def __init__(self, **kwargs):
        self.log = kwargs

    def __call__(self, text: str) -> List[str]:
        pass


class BasicTokenizer(Tokenizer):
    """Basic tokenizer which tokenizes by splitting tokens by blank spaces."""

    def __call__(self, text: str) -> List[str]:
        """Tokenize text to list of tokens.

        Args:
            text (str): Text to be tokenized.

        Returns: List[str]
        """
        tokens = text.split()
        return tokens if tokens is not None else []
