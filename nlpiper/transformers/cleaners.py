"""Cleaner Module."""

import re
from string import punctuation
from unicodedata import normalize, combining

from nlpiper.core.document import Document
from nlpiper.transformers import BaseTransformer

__all__ = ["RemoveUrl", "RemoveEmail", "RemoveNumber", "RemovePunctuation", "RemoveHTML", "StripAccents"]


class Cleaner(BaseTransformer):
    """Abstract class to Cleaners."""

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


class RemoveUrl(Cleaner):
    """Remove URLs."""

    def __call__(self, doc: Document) -> Document:
        """Remove URLs from a document.

        Args:
            doc (Document): document to be cleaned.

        Returns: Document
        """
        super()._validate_document(doc)

        doc.cleaned = re.sub(r"http\S+", "", doc.cleaned)
        doc.cleaned = re.sub(r"www\S+", "", doc.cleaned)

        return doc


class RemoveEmail(Cleaner):
    """Remove Emails."""

    def __call__(self, doc: Document) -> Document:
        """Remove emails from a document.

        Args:
            doc (Document): document to be cleaned.

        Returns: Document
        """
        super()._validate_document(doc)

        doc.cleaned = re.sub(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", "", doc.cleaned)

        return doc


class RemoveNumber(Cleaner):
    """Remove Numbers."""

    def __call__(self, doc: Document) -> Document:
        """Remove numbers from a document.

        Args:
            doc (Document): document to be cleaned.

        Returns: Document
        """
        super()._validate_document(doc)

        doc.cleaned = re.sub(r'[0-9]+', '', doc.cleaned)
        return doc


class RemovePunctuation(Cleaner):
    """Remove Punctuation."""

    def __call__(self, doc: Document) -> Document:
        """Remove punctuation from a document.

        Args:
            doc (Document): document to be cleaned.

        Returns: Document
        """
        super()._validate_document(doc)

        doc.cleaned = doc.cleaned.translate(str.maketrans('', '', punctuation))
        return doc


class RemoveHTML(Cleaner):
    """Remove HTML and XML using BeautifulSoup4."""

    def __init__(self, features: str = "html.parser", *args, **kwargs):
        """Remove HTML and XML.

        Args:
            features (str): Parser used to remove HTML and XML, which could be used the following parsers:
            ```"html.parser"```, ```"lxml"```, ```"lxml-xml"```, ```"xml"```,  ```"html5lib"```,
             for more information about the parser
             go to: https://www.crummy.com/software/BeautifulSoup/bs4/doc/#installing-a-parser
            *args: See the docs at https://www.crummy.com/software/BeautifulSoup/bs4/doc/ for more information.
            **kwargs: See the docs at https://www.crummy.com/software/BeautifulSoup/bs4/doc/ for more information.
        """
        super().__init__(features=features, *args, **kwargs)
        try:
            from bs4 import BeautifulSoup
            self.c = BeautifulSoup
            self.args = args
            self.kwargs = kwargs
            self.features = features

        except ImportError:
            print("Please install BeautifulSoup4. "
                  "See the docs at https://www.crummy.com/software/BeautifulSoup/ for more information.")
            raise

    def __call__(self, doc: Document) -> Document:
        """Remove HTML and XML from the document.

        Args:
            doc (Document): document to be cleaned.

        Returns: Document
        """
        super()._validate_document(doc)

        doc.cleaned = self.c(doc.cleaned, features=self.features, *self.args, **self.kwargs).get_text()
        return doc


class StripAccents(Cleaner):
    """Remove accents and character normalization."""

    def __init__(self, mode: str = "unicode"):
        """Strip accents and character normalization from a document.

        Args:
            mode (str): Available methods: `ascii` and `unicode`. The first method is faster and only works for
            characters that have an direct ASCII mapping. The second is sightly slower but works in any characters.
            (Default: `unicode`)

        Warning: mode `ascii` is only suited for languages that have a direct
        transliteration to ASCII symbols.
        """
        super().__init__(mode=mode)
        self.mode = mode
        assert mode in ['unicode', 'ascii'], (f"{mode} is not implemented. "
                                              f"The only available modes are: 'unicode' and 'ascii'.")

    def __call__(self, doc: Document) -> Document:
        """Strip accents from the document.

        Args:
            doc (Document): document to be cleaned.

        Returns: Document
        """
        super()._validate_document(doc)

        doc.cleaned = (self._strip_accents_unicode(doc.cleaned) if self.mode == 'unicode'
                       else self._strip_accents_ascii(doc.cleaned))
        return doc

    @staticmethod
    def _strip_accents_unicode(text):
        """Strip accents using unicode method.

        Args:
            text (str): Text to be stripped.

        Returns: str
        """
        try:
            # If `s` is ASCII-compatible, then it does not contain any accented
            # characters and we can avoid an expensive list comprehension
            text.encode("ASCII", errors="strict")
            return text
        except UnicodeEncodeError:
            normalized = normalize('NFKD', text)
            return ''.join([c for c in normalized if not combining(c)])

    @staticmethod
    def _strip_accents_ascii(text):
        """Strip accents using ASCII method.

        This will transform accentuated unicode symbols into ASCII or nothing.

        Warning: this is only suited for languages that have a direct
        transliteration to ASCII symbols.

        Args:
            text (str): Text to be stripped.

        Returns: str
        """
        nkfd_form = normalize('NFKD', text)
        return nkfd_form.encode('ASCII', 'ignore').decode('ASCII')
