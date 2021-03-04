"""Compose Module."""

from typing import Any, List, Union
from collections import OrderedDict


class Compose:
    """Pipeline for process text."""

    def __init__(self, transforms: List[Any]) -> None:
        """Pipeline for process text.

        Args:
            transforms (List[Any]): List of callable objects with implemented method ```__call__```.
        """
        self.transforms = transforms
        self.log = self._log()

    def _log(self):
        d = OrderedDict()
        for transform in self.transforms:
            d[str(transform.__class__)] = transform.log
        return d

    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """Process Text.

        Args:
            text (Union[str, List[str]]): Text to be processed.

        Returns: Union[str, List[str]]
        """
        for t in self.transforms:
            text = t(text)
        return text
