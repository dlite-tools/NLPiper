"""Normalizer Module."""

from string import punctuation

from nlpiper.core.document import Document
from nlpiper.transformers import BaseTransformer

__all__ = ["CaseTokens", "RemovePunctuation"]


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

    def __init__(self, language: str = "english", lower_case: bool = True):
        """Remove stop words.

        When removing stop words, the token will be replaced by an empty string, `""` if is a stop word.

        Args:
            language (str): Language chosen to remove stop words.
            lower_case (bool): If is case sensitive or not. If `True` the token will be case lowered to compare with
            stop words, e.g. token: "The" -> "the" which is within stop words, otherwise, it will not be considered a
            stop word.
        """
        super().__init__(language=language)
        self.case = "lower" if lower_case else "__str__"
        try:
            import nltk
            self.stopwords = nltk.corpus.stopwords.words(language)

        except ImportError:
            print("Please install NLTK. "
                  "See the docs at https://www.nltk.org/install.html for more information.")
            raise

    def __call__(self, tokens: Union[List[List[str]], Document]) -> Document:
        """Remove Stop Words.

        Args:
            tokens (Union[List[List[str]], Document]): List of tokens to be normalized.

        Returns: Document
        """
        if isinstance(tokens, list):
            phrases = [" ".join(phrase) for phrase in tokens]
            doc = Document(" ".join(phrases))
            doc.phrases = phrases
            doc.tokens = [[Token(token) for token in phrase] for phrase in tokens]
        else:
            doc = tokens

        for phrase in doc.tokens:
            for token in phrase:
                if token.processed is None:
                    token.processed = "" if getattr(token.original, self.case)() in self.stopwords else token.original
                else:
                    token.processed = "" if getattr(token.processed, self.case)() in self.stopwords else token.processed

        return doc
