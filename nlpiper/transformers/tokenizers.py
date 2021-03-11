"""Tokenizer Module."""

from typing import Union

from nlpiper.core.document import Document, Token

__all__ = ["BasicTokenizer", "MosesTokenizer"]


class Tokenizer:
    """Abstract class to Tokenizers."""

    def __init__(self, *args, **kwargs):
        args = {"args": list(args)} if len(args) != 0 else {}
        self.log = {**kwargs, **args}

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
    """SacreMoses tokenizer.

    Transformer to tokenize text using Sacremoses, https://github.com/alvations/sacremoses
    """

    def __init__(self, *args, **kwargs):
        """SacreMoses tokenizer.

        Args:
            *args: See the docs at https://github.com/alvations/sacremoses for more information.
            **kwargs: See the docs at https://github.com/alvations/sacremoses for more information.
        """
        super().__init__(*args, **kwargs)
        try:
            from sacremoses import MosesTokenizer
            self.t = MosesTokenizer(*args, **kwargs)

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
