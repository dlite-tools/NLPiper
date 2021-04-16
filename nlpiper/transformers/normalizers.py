"""Normalizer Module."""

from string import punctuation
from typing import Optional

from nlpiper.core.document import Document
from nlpiper.transformers import (
    BaseTransformer,
    TransformersType,
    add_step,
    validate
)
from nlpiper.logger import log

__all__ = [
    "CaseTokens",
    "RemovePunctuation",
    "RemoveStopWords",
    "SpellCheck",
]


class CaseTokens(BaseTransformer):
    """Uppercase or Lowercase tokens."""

    def __init__(self, mode='lower'):
        """Case tokens.

        Args:
            mode (str): Mode can be ```'lower'``` or ```'upper'```, lowering or upper casing the letters respectively.
        """
        if mode not in ('upper', 'lower'):
            raise ValueError(f'{mode} mode is not available, it can only be "upper" or "lower".')

        super().__init__(mode=mode)
        self.mode = mode

    @validate(TransformersType.NORMALIZERS)
    @add_step
    def __call__(self, doc: Document, inplace: bool = False) -> Optional[Document]:
        """Uppercase or Lowercase tokens.

        Args:
            doc (Document): Document to be normalized.
            inplace (bool): if True will return a new doc object,
                            otherwise will change the object passed as parameter.

        Returns: Document
        """
        d = doc if inplace else doc._deepcopy()

        for token in d.tokens:
            token.cleaned = getattr(token.cleaned, self.mode)()

        return None if inplace else d


class RemovePunctuation(BaseTransformer):
    """Remove Punctuation."""

    @validate(TransformersType.NORMALIZERS)
    @add_step
    def __call__(self, doc: Document, inplace: bool = False) -> Optional[Document]:
        """Remove punctuation.

        Args:
            doc (Document): Document to be normalized.
            inplace (bool): if True will return a new doc object,
                            otherwise will change the object passed as parameter.

        Returns: Document
        """
        d = doc if inplace else doc._deepcopy()

        for token in d.tokens:
            token.cleaned = token.cleaned.translate(str.maketrans('', '', punctuation))

        return None if inplace else d


class RemoveStopWords(BaseTransformer):
    """Remove Stop Words."""

    def __init__(self, language: str = "english", case_sensitive: bool = True):
        """Remove stop words.

        When removing stop words, the token will be replaced by an empty string, `""` if is a stop word.

        Args:
            language (str): Language chosen to remove stop words.
            case_sensitive (bool): When True, the detection of stop words will be case sensitive, e.g. 'This' is a stop
            word, however, since 'T' is upper case will not be considered as a stop word, otherwise, will be considered
            as a stop word and replaced by an empty string, "".
        """
        super().__init__(language=language, case_sensitive=case_sensitive)
        self.case_sensitive = "__str__" if case_sensitive else "lower"
        try:
            import nltk
            nltk.download("stopwords")
            self.stopwords = nltk.corpus.stopwords.words(language)

        except ImportError:
            log.error("Please install NLTK. "
                      "See the docs at https://www.nltk.org/install.html for more information.")
            raise

    @validate(TransformersType.NORMALIZERS)
    @add_step
    def __call__(self, doc: Document, inplace: bool = False) -> Optional[Document]:
        """Remove Stop Words.

        Args:
            doc (Document): Document to be normalized.
            inplace (bool): if True will return a new doc object,
                            otherwise will change the object passed as parameter.

        Returns: Document
        """
        d = doc if inplace else doc._deepcopy()

        for token in d.tokens:
            token.cleaned = "" if getattr(token.cleaned, self.case_sensitive)() in self.stopwords else token.cleaned

        return None if inplace else d


class SpellCheck(BaseTransformer):
    """Spellcheck the tokens."""

    def __init__(self, language: str = "en_GB", max_distance: Optional[int] = None, *args, **kwargs):
        """Spellcheck tokens.

        Args:
            language (str): By default you have the following dictionaries available `'en_AU'`, `'en_CA'`, `'en_GB'`,
             `'en_NZ'`, `'en_US'`, `'en_ZA'`, Default(`"en_GB"`).
            max_distance (Optional[int]): Default(`None`)
        """
        super().__init__(language=language, max_distance=max_distance, *args, **kwargs)
        self.max_distance = max_distance
        try:
            from hunspell import Hunspell
            self.h = Hunspell(lang=language, *args, **kwargs)

        except ImportError:
            log.error("Please install cyhunspell. "
                      "See the docs at https://pypi.org/project/cyhunspell/ for more information.")
            raise

        if max_distance:
            try:
                from nltk.metrics.distance import edit_distance
                self.edit_distance = edit_distance

            except ImportError:
                log.error("Please install NLTK. "
                          "See the docs at https://www.nltk.org/install.html for more information.")
                raise

    @validate(TransformersType.NORMALIZERS)
    @add_step
    def __call__(self, doc: Document, inplace: bool = False) -> Optional[Document]:
        """

        Args:
            doc (Document): Document to be normalized.
            inplace (bool): if True will return a new doc object,
                            otherwise will change the object passed as parameter.

        Returns: Document
        """
        d = doc if inplace else doc._deepcopy()

        def suggest(cleaned: str):
            if self.max_distance:
                suggestions = self.h.suggest(cleaned)

                distances = [self.edit_distance(cleaned, s) for s in suggestions]
                min_distance = min(distances)

                return suggestions[distances.index(min_distance)] if self.max_distance >= min_distance else cleaned
            else:
                return ''

        for token in d.tokens:
            token.cleaned = token.cleaned if self.h.spell(token.cleaned) else suggest(token.cleaned)

        return None if inplace else d
