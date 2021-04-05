"""Compose Module."""
from typing import (
    Optional,
    List
)

from nlpiper.core.document import Document
from nlpiper.transformers import BaseTransformer
from nlpiper.logger import log

# Needed for create_from_steps method
from nlpiper.transformers.cleaners import *  # noqa: F401, F403
from nlpiper.transformers.normalizers import *  # noqa: F401, F403
from nlpiper.transformers.tokenizers import *  # noqa: F401, F403


class Compose:
    """Pipeline for process document."""

    def __init__(self, transformers: List[BaseTransformer]) -> None:
        """Pipeline for process text.

        Args:
            transformers (List[BaseTransformer]): List of callable objects with implemented method ```__call__```.
        """
        self.transformers = transformers
        log.info("[Created] %s", repr(self))

    @classmethod
    def create_from_steps(self, steps: List[str]):
        try:
            transformers = [eval(step) for step in steps]
        except NameError as e:
            log.error("Unable to create Compose object from steps: %s", steps)
            raise e

        return Compose(transformers)

    def __repr__(self) -> str:
        params = ', '.join([repr(t) for t in self.transformers])
        return "%s([%s])" % (self.__class__.__name__, params)

    def __call__(self, doc: Document, inplace: bool = False) -> Optional[Document]:
        """Process document with transformers pipeline.

        Args:
            doc (Document): Document object to be processed.
            inplace (bool): if True will return a new doc object,
                            otherwise will change the object passed as parameter.

        Returns: Document
        """
        d = doc if inplace else doc._deepcopy()

        for t in self.transformers:
            t(d, True)

        return None if inplace else d
