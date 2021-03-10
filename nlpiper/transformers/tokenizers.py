"""Tokenizer Module."""

from typing import Union

from nlpiper.core.document import Document, Token

__all__ = ["BasicTokenizer", "MosesTokenizer"]


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


class MosesTokenizer(Tokenizer):
    """SacreMoses tokenizer."""

    def __init__(self, **kwargs):
        """SacreMoses tokenizer.

        Args:
            **kwargs: See the docs at https://github.com/alvations/sacremoses for more information.
        """
        super().__init__(**kwargs)
        try:
            from sacremoses import MosesTokenizer
            self.t = MosesTokenizer()

        except ImportError:
            print("Please install SacreMoses. "
                  "See the docs at https://github.com/alvations/sacremoses for more information.")
            raise

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
            phrases = [self.t.tokenize(phrase) if phrase is not None else [] for phrase in doc.phrases]

        else:
            phrases = [self.t.tokenize(doc.cleaned) if doc.cleaned is not None else self.t.tokenize(doc.text)]

        doc.tokens = [[Token(token) for token in phrase] for phrase in phrases]

        return doc
