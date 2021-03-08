"""Tokenizer Module."""

from typing import Union

from nlpiper.core.document import Document, Token

__all__ = ["BasicTokenizer"]


class Tokenizer:
    """Abstract class to Tokenizers."""

    def __init__(self, **kwargs):
        self.log = kwargs

    def __call__(self, text: Union[str, Document]) -> Document:
        raise NotImplementedError


class BasicTokenizer(Tokenizer):
    """Basic tokenizer which tokenizes by splitting tokens by blank spaces."""

    def __call__(self, text: Union[str, Document]) -> Document:
        """Tokenize text to list of tokens.

        Args:
            text (Union[str, Document]): Text to be tokenized.

        Returns: Document
        """
        if isinstance(text, str):
            doc = Document(text)
        else:
            doc = text

        if doc.phrases:
            phrases = [phrase.split() if phrase is not None else [] for phrase in doc.phrases]

        else:
            phrases = [doc.cleaned.split() if doc.cleaned is not None else doc.text.split()]

        doc.tokens = [[Token(token) for token in phrase] for phrase in phrases]

        return doc
