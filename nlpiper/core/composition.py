"""Compose Module."""

from typing import Callable, List

from nlpiper.core.document import Document
from nlpiper.logger import log


class Compose:
    """Pipeline for process document."""

    def __init__(self, transformers: List[Callable]) -> None:
        """Pipeline for process text.

        Args:
            transformers (List[Callable]): List of callable objects with implemented method ```__call__```.
        """
        self.transformers = transformers
        log.info("[Created] Compose(%s)", ', '.join([repr(t) for t in transformers]))

    def __call__(self, doc: Document) -> Document:
        """Process document with transformers pipeline.

        Args:
            doc (Document): Document object to be processed.

        Returns: Document
        """
        for t in self.transformers:
            doc = t(doc)
        return doc
