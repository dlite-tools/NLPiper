"""Tokenizer Module."""

from typing import Optional
from nlpiper.core.document import (
    Document,
    Token
)
from nlpiper.logger import log
from nlpiper.transformers import (
    BaseTransformer,
    TransformersType,
    add_step,
    validate
)


__all__ = [
    "BasicTokenizer",
    "MosesTokenizer",
    "StanzaTokenizer"
]


class BasicTokenizer(BaseTransformer):
    """Basic tokenizer which tokenizes a document by splitting tokens by its blank spaces."""

    @validate(TransformersType.TOKENIZERS)
    @add_step
    def __call__(self, doc: Document, inplace: bool = False) -> Optional[Document]:
        """Tokenize the document in a list of tokens.

        Args:
            doc (Document): Text to be tokenized.
            inplace (bool): if False will return a new doc object,
                            otherwise will change the object passed as parameter.

        Returns: Document
        """
        d = doc if inplace else doc._deepcopy()

        d.tokens = [Token(token) for token in d.cleaned.split()]

        return None if inplace else d


class MosesTokenizer(BaseTransformer):
    """SacreMoses tokenizer.

    Transformer to tokenize a Document using Sacremoses, https://github.com/alvations/sacremoses
    """

    def __init__(self, *args, **kwargs):
        """SacreMoses tokenizer.

        Args:
            *args: See the docs at https://github.com/alvations/sacremoses for more information.
            **kwargs: See the docs at https://github.com/alvations/sacremoses for more information.
        """
        super().__init__(*args, **kwargs)
        try:
            from sacremoses import MosesTokenizer
            self.t = MosesTokenizer(*args, **kwargs)

        except ImportError:
            log.error("Please install SacreMoses. "
                      "See the docs at https://github.com/alvations/sacremoses for more information.")
            raise

    @validate(TransformersType.TOKENIZERS)
    @add_step
    def __call__(self, doc: Document, inplace: bool = False) -> Optional[Document]:
        """Tokenize the document in a list of tokens.

        Args:
            doc (Document): Document to be tokenized.
            inplace (bool): if False will return a new doc object,
                            otherwise will change the object passed as parameter.

        Returns: Document
        """
        d = doc if inplace else doc._deepcopy()

        d.tokens = [Token(token) for token in self.t.tokenize(d.cleaned)]

        return None if inplace else d


class StanzaTokenizer(BaseTransformer):
    """Stanza tokenizer.

    Transformer to tokenize a Document using stanza, https://github.com/stanfordnlp/stanza
    """

    def __init__(self, language: str = 'en', *args, **kwargs):
        """Stanza tokenizer.

        Args:
            language (str): document main language.
            *args: See the docs at https://stanfordnlp.github.io/stanza/tokenize.html add for more information.
            **kwargs: See the docs at https://stanfordnlp.github.io/stanza/tokenize.html add for more information.
        """
        super().__init__(language=language, *args, **kwargs)
        try:
            import stanza
            from stanza import Pipeline
            stanza.download(language)
            self.p = Pipeline(lang=language, processors='tokenize', tokenize_pretokenized=False, *args,
                              **kwargs)

        except ImportError:
            log.error("Please install Stanza. "
                      "See the docs at https://github.com/stanfordnlp/stanza for more information.")
            raise

    @validate(TransformersType.TOKENIZERS)
    @add_step
    def __call__(self, doc: Document, inplace: bool = False) -> Optional[Document]:
        """Tokenize the document in a list of tokens.

        Args:
            doc (Document): Document to be tokenized.
            inplace (bool): if False will return a new doc object,
                            otherwise will change the object passed as parameter.

        Returns: Document
        """
        d = doc if inplace else doc._deepcopy()

        d.tokens = [Token(word.text) for sentence in self.p(doc.cleaned).sentences for word in sentence.words]

        return None if inplace else d
