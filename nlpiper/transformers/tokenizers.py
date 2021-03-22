"""Tokenizer Module."""

from nlpiper.core.document import Document, Token
from nlpiper.transformers import BaseTransformer

__all__ = ["BasicTokenizer", "MosesTokenizer"]


class Tokenizer(BaseTransformer):
    """Abstract class to Tokenizers."""

    def _validate_document(self, doc: Document):
        """Validate if document is ready to be processed.

        Args:
            doc (Document): document to be tokenized.

        Raises:
            TypeError: if doc is not a Document.
        """
        if not isinstance(doc, Document):
            raise TypeError("Argument doc is not of type Document")

        if doc.cleaned is None:
            doc.cleaned = doc.original

        if doc.phrases is None:
            doc.phrases = [doc.cleaned]


class BasicTokenizer(Tokenizer):
    """Basic tokenizer which tokenizes a document by splitting tokens by its blank spaces."""

    def __call__(self, doc: Document) -> Document:
        """Tokenize the document in a list of tokens.

        Args:
            doc (Document): Text to be tokenized.

        Returns: Document
        """
        super()._validate_document(doc)

        doc.tokens = [[Token(original=token) for token in phrase.split()] for phrase in doc.phrases]

        return doc


class MosesTokenizer(Tokenizer):
    """SacreMoses tokenizer.

    Transformer to tokenize a Document using Sacremoses, https://github.com/alvations/sacremoses
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

    def __call__(self, doc: Document) -> Document:
        """Tokenize the document in a list of tokens.

        Args:
            doc (Document): Document to be tokenized.

        Returns: Document
        """
        super()._validate_document(doc)

        doc.tokens = [[Token(original=token) for token in self.t.tokenize(phrase)] for phrase in doc.phrases]

        return doc
