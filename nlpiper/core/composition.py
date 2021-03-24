"""Compose Module."""

from typing import Callable, List

from nlpiper.core.document import Document
from nlpiper.transformers import BaseTransformer
from nlpiper.logger import log


class Compose:
    """Pipeline for process document."""

    def __init__(self, transformers: List[BaseTransformer]) -> None:
        """Pipeline for process text.

        Args:
            transformers (List[BaseTransformer]): List of callable objects with implemented method ```__call__```.
        """
        self.transformers = transformers
        log.info("[Created] %s", repr(self))

    def __repr__(self) -> str:
        params = ', '.join([repr(t) for t in self.transformers])
        return "%s([%s])" % (self.__class__.__name__, params)

    def __call__(self, doc: Document) -> Document:
        """Process document with transformers pipeline.

        Args:
            doc (Document): Document object to be processed.

        Returns: Document
        """
        for t in self.transformers:
            doc = t(doc)
        return doc
