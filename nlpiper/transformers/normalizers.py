"""Normalizer Module."""

from string import punctuation
from typing import (
    Optional,
    List
)

from nlpiper.core import Document
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
    "VocabularyFilter",
    "SpellCheck",
    "Stemmer"
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
            inplace (bool): if False will return a new doc object,
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
            inplace (bool): if False will return a new doc object,
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
            inplace (bool): if False will return a new doc object,
                            otherwise will change the object passed as parameter.

        Returns: Document
        """
        d = doc if inplace else doc._deepcopy()

        for token in d.tokens:
            token.cleaned = "" if getattr(token.cleaned, self.case_sensitive)() in self.stopwords else token.cleaned

        return None if inplace else d


class VocabularyFilter(BaseTransformer):
    """Only allow tokens from a pre-defined vocabulary."""

    def __init__(self, vocabulary: List[str], case_sensitive: bool = True):
        """Only allow tokens from a pre-defined vocabulary.

        Only accept tokens that are in the vocabulary, otherwise the token will be replace by an empty string, `""`.

        Args:
            vocabulary (str): List of tokens that define the vocabulary.
            case_sensitive (bool): When `True`, the detection of a token in the vocabulary will be case sensitive,
             e.g. `vocab = ['this']`, if `'This'` is a token, since 'T' is upper case and will not be considered as a
             token from the vocabulary and will be replaced by an empty string, `""`, otherwise, will be considered
            as in vocabulary and kept.
        """
        super().__init__(vocabulary=vocabulary, case_sensitive=case_sensitive)
        self.vocab = vocabulary if case_sensitive else [token.lower() for token in vocabulary]
        self.case_sensitive = "__str__" if case_sensitive else "lower"

    @validate(TransformersType.NORMALIZERS)
    @add_step
    def __call__(self, doc: Document, inplace: bool = False) -> Optional[Document]:
        """Remove Stop Words.

        Args:
            doc (Document): Document to be normalized.
            inplace (bool): if `False` will return a new doc object,
                            otherwise will change the object passed as parameter.

        Returns: Document
        """
        d = doc if inplace else doc._deepcopy()

        for token in d.tokens:
            token.cleaned = "" if getattr(token.cleaned, self.case_sensitive)() not in self.vocab else token.cleaned

        return None if inplace else d


class Stemmer(BaseTransformer):
    """Stem tokens."""

    def __init__(self, version: str = 'nltk', language: str = "english", *args, **kwargs):
        """Stem tokens.

        Stemmer currently supports two way to stem the tokens, using NLTK SnowballStemmer or using Hunspell.

        Args:
            version (str): Currently there are two stemmers available: `nltk` and `hunspell`.
            language (str): Available languages for `nltk`: "arabic", "danish", "dutch", "english", "finnish", "french",
             "german", "hungarian", "italian", "norwegian", "porter", "portuguese", "romanian", "russian", "spanish",
             "swedish". (Default: `"english"`) For `hunspell`  by default the following languages are available:
             `'en_AU'`, `'en_CA'`, `'en_GB'`, `'en_NZ'`, `'en_US'`, `'en_ZA'`, however is possible to use other
             dictionaries, for this please check https://pypi.org/project/cyhunspell/
        """
        super().__init__(version=version, language=language, *args, **kwargs)
        if version == 'nltk':
            try:
                import nltk  # noqa: F401
                from nltk.stem.snowball import SnowballStemmer
                self.stemmer = SnowballStemmer(language=language, *args, **kwargs)

            except ImportError:
                log.error("Please install NLTK. "
                          "See the docs at https://www.nltk.org/install.html for more information.")
                raise

        elif version == 'hunspell':
            try:
                from hunspell import Hunspell
                self.stemmer = Hunspell(lang=language, *args, **kwargs)

            except ImportError:
                log.error("Please install cyhunspell. "
                          "See the docs at https://pypi.org/project/cyhunspell/ for more information.")
                raise

        else:
            raise ValueError(f"Currently {repr(version)} is not available."
                             f" You can opt by using 'nltk' or 'hunspell' to stem the tokens.")

    @validate(TransformersType.NORMALIZERS)
    @add_step
    def __call__(self, doc: Document, inplace: bool = False) -> Optional[Document]:
        """Stem tokens.

        Args:
            doc (Document): Document to be normalized.
            inplace (bool): if False will return a new doc object,
                            otherwise will change the object passed as parameter.

        Returns: Document
        """
        d = doc if inplace else doc._deepcopy()

        for token in d.tokens:
            stem = self.stemmer.stem(token.cleaned)
            stem = stem[0] if isinstance(stem, tuple) else stem
            token.cleaned = stem if stem else token.cleaned
            token.stem = stem if stem else token.cleaned

        return None if inplace else d


class SpellCheck(BaseTransformer):
    """Perform Spellcheck on tokens."""

    def __init__(self, language: str = "en_GB", max_distance: Optional[int] = None, *args, **kwargs):
        """Perform Spellcheck on tokens.

        Uses Hunspell spellchecker engine.

        Args:
            language (str): By default the following dictionaries are available: `'en_AU'`, `'en_CA'`, `'en_GB'`,
             `'en_NZ'`, `'en_US'`, `'en_ZA'`, however is possible to use other dictionaries, for this please check
              https://pypi.org/project/cyhunspell/, Default(`"en_GB"`).
            max_distance (Optional[int]): If `None`, the tokens that are not spelt correctly are replaced by
             an empty string, otherwise will maintain the original token. To use `max_distance` is necessary to install
              `nltk` package, which will check if a token is correctly spelt or not, if not than it will check the
              suggested words by Hunspell and calculate the levenshtein distance between the token and the suggestions,
              and will replace the token by the word with the lower distance if is also lower than the `max_distance`,
              otherwise will maintain the original token. Default(`None`)
            args: For further utilities check https://pypi.org/project/cyhunspell/
            kwargs: For further utilities check https://pypi.org/project/cyhunspell/
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
                import nltk  # noqa: F401
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
            inplace (bool): if False will return a new doc object,
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
