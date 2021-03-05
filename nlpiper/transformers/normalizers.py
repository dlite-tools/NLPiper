"""Normalizer Module."""

from string import punctuation
from typing import List, Union

from nlpiper.core.document import Document, Token

__all__ = ["CaseTokens", "RemovePunctuation"]


class Normalizer:
    """Abstract class to Normalizers."""

    def __init__(self, **kwargs):
        self.log = kwargs

    def __call__(self, tokens: Union[List[List[str]], Document]) -> Document:
        pass


class CaseTokens(Normalizer):
    """Lower tokens."""

    def __init__(self, mode='lower'):
        """Lower tokens.

        Args:
            mode (str): Mode can be ```'lower'``` or ```'upper'```, lowering or upper casing the letters respectively.
        """
        super().__init__(mode=mode)
        assert mode in ('upper', 'lower'), f'{mode} mode is not available, it can only be "upper" or "lower".'
        self.mode = mode

    def __call__(self, tokens: Union[List[List[str]], Document]) -> Document:
        """Lower Tokens.

        Args:
            tokens (Union[List[List[str]], Document]): List of tokens to be normalized.

        Returns: Document
        """
        if isinstance(tokens, list):
            phrases = [" ".join(phrase) for phrase in tokens]
            doc = Document(" ".join(phrases))
            doc.phrases = phrases
            doc.tokens = [[Token(token) for token in phrase] for phrase in tokens]
        else:
            doc = tokens

        for phrase in doc.tokens:
            for token in phrase:
                if token.processed is None:
                    token.processed = getattr(token.original, self.mode)()
                else:
                    token.processed = getattr(token.processed, self.mode)()

        return doc


class RemovePunctuation(Normalizer):
    """Remove Punctuation."""

    def __call__(self, tokens: Union[List[List[str]], Document]) -> Document:
        """Remove punctuation.

        Args:
            tokens (Union[List[List[str]], Document]): List of tokens to be normalized.

        Returns: List[str]
        """
        if isinstance(tokens, list):
            phrases = [" ".join(phrase) for phrase in tokens]
            doc = Document(" ".join(phrases))
            doc.phrases = phrases
            doc.tokens = [[Token(token) for token in phrase] for phrase in tokens]
        else:
            doc = tokens

        for phrase in doc.tokens:
            for token in phrase:
                if token.processed is None:
                    token.processed = token.original.translate(str.maketrans('', '', punctuation))
                else:
                    token.processed = token.processed.translate(str.maketrans('', '', punctuation))

        return doc
