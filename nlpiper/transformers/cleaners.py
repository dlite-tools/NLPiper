"""Cleaner Module."""

import re
from string import punctuation
from typing import Union

from nlpiper.core.document import Document

__all__ = ["RemoveEmail", "RemoveNumber", "RemoveUrl", "RemovePunctuation"]


class Cleaner:
    """Abstract class to Cleaners."""

    def __init__(self, *args, **kwargs):
        args = {"args": list(args)} if len(args) != 0 else {}
        self.log = {**kwargs, **args}

    def __call__(self, text: Union[str, Document]) -> Document:
        raise NotImplementedError


class RemoveUrl(Cleaner):
    """Remove URLs."""

    def __call__(self, text: Union[str, Document]) -> Document:
        """Remove text URLs.

        Args:
            text (Union[str, Document]): text to be cleaned.

        Returns: Document
        """
        if isinstance(text, str):
            doc = Document(original=text)
        else:
            doc = text

        if doc.cleaned is None:
            doc.cleaned = doc.original

        doc.cleaned = re.sub(r"http\S+", "", doc.cleaned)
        doc.cleaned = re.sub(r"www\S+", "", doc.cleaned)

        return doc


class RemoveEmail(Cleaner):
    """Remove Emails."""

    def __call__(self, text: Union[str, Document]) -> Document:
        """Remove text Emails.

        Args:
            text (Union[str, Document]): text to be cleaned.

        Returns: Document
        """
        if isinstance(text, str):
            doc = Document(original=text)
        else:
            doc = text

        if doc.cleaned is None:
            doc.cleaned = doc.original

        doc.cleaned = re.sub(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", "", doc.cleaned)

        return doc


class RemoveNumber(Cleaner):
    """Remove Numbers."""

    def __call__(self, text: Union[str, Document]) -> Document:
        """Remove text Numbers.

        Args:
            text (Union[str, Document]): text to be cleaned.

        Returns: Document
        """
        if isinstance(text, str):
            doc = Document(original=text)
        else:
            doc = text

        if doc.cleaned is None:
            doc.cleaned = doc.original

        doc.cleaned = re.sub(r'[0-9]+', '', doc.cleaned)
        return doc


class RemovePunctuation(Cleaner):
    """Remove Punctuation."""

    def __call__(self, text: Union[str, Document]) -> Document:
        """Remove Punctuation from text.

        Args:
            text (Union[str, Document]): text to be cleaned.

        Returns: Document
        """
        if isinstance(text, str):
            doc = Document(original=text)
        else:
            doc = text

        if doc.cleaned is None:
            doc.cleaned = doc.original

        doc.cleaned = doc.cleaned.translate(str.maketrans('', '', punctuation))
        return doc


class RemoveHTML(Cleaner):
    """Remove HTML and XML using BeautifulSoup4."""

    def __init__(self, features: str = "html.parser", *args, **kwargs):
        """Remove HTML and XML.

        Args:
            features (str): Parser used to remove HTML and XML, which could be used the following parsers:
            ```"html.parser"```, ```"lxml"```, ```"lxml-xml"```, ```"xml"```,  ```"html5lib"```,
             for more information about the parser go to: https://www.crummy.com/software/BeautifulSoup/bs4/doc/#installing-a-parser
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

    def __call__(self, text: Union[str, Document]) -> Document:
        """Remove HTML and XML from text.

        Args:
            text (Union[str, Document]): text to be cleaned.

        Returns: Document
        """
        if isinstance(text, str):
            doc = Document(original=text)
        else:
            doc = text

        if doc.cleaned is None:
            doc.cleaned = doc.original

        doc.cleaned = self.c(doc.cleaned, features=self.features, *self.args, **self.kwargs).get_text()
        return doc
