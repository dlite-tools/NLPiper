"""Cleaner Module."""

import re
from string import punctuation


__all__ = ["RemoveEmail", "RemoveNumber", "RemoveUrl", "RemovePunctuation"]


class Cleaner:
    """Abstract class to Cleaners."""

    def __init__(self, **kwargs):
        self.log = kwargs

    def __call__(self, text: str) -> str:
        pass


class RemoveUrl(Cleaner):
    """Remove URLs."""

    def __call__(self, text: str) -> str:
        """Remove text URLs.

        Args:
            text (str): text to be cleaned.

        Returns: str
        """
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"www\S+", "", text)
        return text


class RemoveEmail(Cleaner):
    """Remove Emails."""

    def __call__(self, text: str) -> str:
        """Remove text Emails.

        Args:
            text (str): text to be cleaned.

        Returns: str
        """
        return re.sub(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", "", text)


class RemoveNumber(Cleaner):
    """Remove Numbers."""

    def __call__(self, text: str) -> str:
        """Remove text Numbers.

        Args:
            text (str): text to be cleaned.

        Returns: str
        """
        return re.sub(r'[0-9]+', '', text)


class RemovePunctuation(Cleaner):
    """Remove Punctuation."""

    def __call__(self, text: str) -> str:
        """Remove Punctuation from text.

        Args:
            text (str): text to be cleaned.

        Returns: str

        """
        return text.translate(str.maketrans('', '', punctuation))
