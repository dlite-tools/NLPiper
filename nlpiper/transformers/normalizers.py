"""Normalizer Module."""

from string import punctuation
from typing import List


__all__ = ["CaseTokens", "RemovePunctuation"]


class Normalizer:
    """Abstract class to Normalizers."""

    def __init__(self, **kwargs):
        self.log = kwargs

    def __call__(self, tokens: List[str]) -> List[str]:
        pass


class CaseTokens(Normalizer):
    """Lower tokens."""

    def __init__(self, mode='lower'):
        """

        Args:
            mode (str): Mode can be ```'lower'``` or ```'upper'```, lowering or upper casing the letters respectively.
        """
        super().__init__(mode=mode)
        assert mode in ('upper', 'lower'), f'{mode} mode is not available, it can only be "upper" or "lower".'
        self.mode = mode

    def __call__(self, tokens: List[str]) -> List[str]:
        """Lower Tokens.

        Args:
            tokens (List[str]): List of tokens to be normalized.

        Returns: List[str]
        """
        return [getattr(token, self.mode)() for token in tokens]


class RemovePunctuation(Normalizer):
    """Remove Punctuation."""

    def __call__(self, tokens: List[str]) -> List[str]:
        """Remove punctuation.

        Args:
            tokens (List[str]): List of tokens to be normalized.

        Returns: List[str]
        """
        return [token.translate(str.maketrans('', '', punctuation)) for token in tokens
                if len(token.translate(str.maketrans('', '', punctuation))) > 0]
