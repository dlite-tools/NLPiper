"""Normalizer Module."""

from string import punctuation

from nlpiper.core.document import Document
from nlpiper.transformers import BaseTransformer

__all__ = ["CaseTokens", "RemovePunctuation", "RemoveStopWords"]


class Normalizer(BaseTransformer):
    """Abstract class to Normalizers."""

    def _validate_document(self, doc: Document):
        """Validate if document is ready to be processed.

        Args:
            doc (Document): document to be cleaned.

        Raises:
            TypeError: if doc is not a Document.
        """
        if not isinstance(doc, Document):
            raise TypeError("Argument doc is not of type Document")

        if doc.cleaned is None:
            doc.cleaned = doc.original

        if doc.phrases is None:
            doc.phrases = doc.cleaned

        if doc.tokens is None:
            raise TypeError("Document does not contain tokens.")

        for phrase in doc.tokens:
            for token in phrase:
                if token.cleaned is None:
                    token.cleaned = token.original


class CaseTokens(Normalizer):
    """Lower tokens."""

    def __init__(self, mode='lower'):
        """Lower tokens.

        Args:
            mode (str): Mode can be ```'lower'``` or ```'upper'```, lowering or upper casing the letters respectively.
        """
        super().__init__(mode=mode)
        assert mode in ('upper', 'lower'), f'{mode} mode is not available, it can only be "upper" or "lower".'
        self.mode = mode

    def __call__(self, doc: Document) -> Document:
        """Lower Tokens.

        Args:
            doc (Document): Document to be normalized.

        Returns: Document
        """
        super()._validate_document(doc)

        for phrase in doc.tokens:
            for token in phrase:
                token.cleaned = getattr(token.cleaned, self.mode)()
        return doc


class RemovePunctuation(Normalizer):
    """Remove Punctuation."""

    def __call__(self, doc: Document) -> Document:
        """Remove punctuation.

        Args:
            doc (Document): Document to be normalized.

        Returns: Document
        """
        super()._validate_document(doc)

        for phrase in doc.tokens:
            for token in phrase:
                token.cleaned = token.cleaned.translate(str.maketrans('', '', punctuation))
        return doc


class RemoveStopWords(Normalizer):
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
        super().__init__(language=language)
        self.case_sensitive = "__str__" if case_sensitive else "lower"
        try:
            import nltk
            nltk.download("stopwords")
            self.stopwords = nltk.corpus.stopwords.words(language)

        except ImportError:
            print("Please install NLTK. "
                  "See the docs at https://www.nltk.org/install.html for more information.")
            raise

    def __call__(self, doc: Document) -> Document:
        """Remove Stop Words.

        Args:
            doc (Document): Document to be normalized.

        Returns: Document
        """
        super()._validate_document(doc)

        for phrase in doc.tokens:
            for token in phrase:
                token.cleaned = "" if getattr(token.cleaned, self.case_sensitive)() in self.stopwords else token.cleaned

        return doc
