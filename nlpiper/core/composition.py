"""Compose Module."""

from typing import Callable, List, Union
from collections import OrderedDict

from nlpiper.core.document import Document


class Compose:
    """Pipeline for process text."""

    def __init__(self, transforms: List[Callable]) -> None:
        """Pipeline for process text.

        Args:
            transforms (List[Any]): List of callable objects with implemented method ```__call__```.
        """
        self.transforms = transforms

    def __call__(self, text: Union[str, List[List[str]], Document]) -> Document:
        """Process Text.

        Args:
            text (Union[str, List[List[str]], Document]): Text to be processed.

        Returns: Document
        """
        for t in self.transforms:
            text = t(text)
        return text
