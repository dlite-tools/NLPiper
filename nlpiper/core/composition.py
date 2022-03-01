"""Compose Module."""
from typing import (
    Optional,
    List
)

from nlpiper.core import Document
from nlpiper.transformers import BaseTransformer
from nlpiper.logger import log

# Needed for create_from_steps method (eval instruction)
from nlpiper.transformers.cleaners import *  # noqa: F401, F403 (flake8 ignore)
from nlpiper.transformers.normalizers import *  # noqa: F401, F403 (flake8 ignore)
from nlpiper.transformers.tokenizers import *  # noqa: F401, F403 (flake8 ignore)
from nlpiper.transformers.embeddings import *  # noqa: F401, F403 (flake8 ignore)


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
    def create_from_steps(cls, steps: List[str]):
        """Create a Compose instance from a list of steps.

        Args:
            steps (List[str]): List of steps applied on a document.

        Returns: Compose
        """
        try:
            transformers = [eval(step) for step in steps]
        except NameError as e:
            log.error("Unable to create Compose object from steps: %s", steps)
            raise e

        return Compose(transformers)

    @classmethod
    def rollback_document(cls, doc: Document, num_steps: int = 1) -> Document:
        """Rollback the steps applied to a document.

        The method will return a new document with the steps applied to the rollback point.

        Args:
            doc (Document): Document instance that will have the steps rolled back.
            num_steps (int, optional): Number of steps to rollback, by default 1.

        Returns: Document
        """
        if len(doc.steps) == 0:
            raise ValueError("Document must have steps to rollback")

        if not (0 < num_steps <= len(doc.steps)):
            raise ValueError(f"Number of steps to rollback must be between 1 and {len(doc.steps)} steps")

        out = Document(doc.original)

        steps = cls.create_from_steps(doc.steps[:-num_steps])

        return steps(out, inplace=False)

    def __repr__(self) -> str:
        params = ', '.join([repr(t) for t in self.transformers])
        return "%s([%s])" % (self.__class__.__name__, params)

    def __call__(self, doc: Document, inplace: bool = False) -> Optional[Document]:
        """Process document with transformers pipeline.

        Args:
            doc (Document): Document object to be processed.
            inplace (bool): if False will return a new doc object,
                            otherwise will change the object passed as parameter.

        Returns: Document
        """
        d = doc if inplace else doc._deepcopy()

        for t in self.transformers:
            t(d, True)

        return None if inplace else d
