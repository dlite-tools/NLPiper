"""Normalizer Module."""

from string import punctuation
from typing import List, Union

from nlpiper.core.document import Document, Token
from nlpiper.transformers import BaseTransformer

__all__ = ["CaseTokens", "RemovePunctuation"]


class Normalizer(BaseTransformer):
    """Abstract class to Normalizers."""

    def _validate_document(self, doc: Document):
        """Validate if document is ready to be processed.

        Args:
            doc (Document): document to be cleaned.

        Raises:
            TypeError: if doc is not a Document.
        """
        if not isinstance(doc, Document):
            raise TypeError("Argument doc is not of type Document")

        if doc.cleaned is None:
            doc.cleaned = doc.original

        if doc.phrases is None:
            doc.phrases = doc.cleaned

        if doc.tokens is None:
            raise TypeError("Document does not contain tokens.")

        for phrase in doc.tokens:
            for token in phrase:
                if token.cleaned is None:
                    token.cleaned = token.original


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

    def __call__(self, doc: Document) -> Document:
        """Lower Tokens.

        Args:
            doc (Document): List of tokens to be normalized.

        Returns: Document
        """
        super()._validate_document(doc)

        for phrase in doc.tokens:
            for token in phrase:
                token.cleaned = getattr(token.cleaned, self.mode)()
        return doc


class RemovePunctuation(Normalizer):
    """Remove Punctuation."""

    def __call__(self, doc: Document) -> Document:
        """Remove punctuation.

        Args:
            tokens (Union[List[List[str]], Document]): List of tokens to be normalized.

        Returns: List[str]
        """
        super()._validate_document(doc)

        for phrase in doc.tokens:
            for token in phrase:
                token.cleaned = token.cleaned.translate(str.maketrans('', '', punctuation))
        return doc
