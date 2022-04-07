"""Embeddings Module."""

from typing import (
    Any,
    Optional
)

from nlpiper.core.document import Document
from nlpiper.transformers.base import (
    BaseTransformer,
    TransformersType,
    add_step,
    validate
)
from nlpiper.logger import log

__all__ = [
    "GensimEmbeddings",
    "TorchTextEmbeddings"
]


class GensimEmbeddings(BaseTransformer):
    """Gensim Embedding extraction.

    Callable arguments:

    Args:
        doc (Document): Document to extract embeddings.
        inplace (bool): if False will return a new doc object,
            otherwise will change the object passed as parameter.

    Returns:
        Document with Gensim Embedding or None if `inplace=True`.

    Example:
        >>> import gensim.downloader
        >>> from nlpiper.transformers.embeddings import GensimEmbeddings
        >>> from nlpiper.core.document import Document
        >>> from nlpiper.transformers.tokenizers import BasicTokenizer
        >>> glove_vectors = gensim.downloader.load('glove-twitter-25')
        >>> doc = Document('Test random stuff.')
        >>> t = BasicTokenizer()
        >>> t(doc, inplace=True)
        >>> e = GensimEmbeddings(glove_vectors)
        >>> type(e(doc).embedded)
        <class 'numpy.ndarray'>
    """

    def __init__(self, keyed_vectors: Any, apply_doc: str = 'mean'):
        """Gensim Embedding extraction.

        Args:
            keyed_vectors (Any): Gensim model based on keyedVectors,
                see more in: https://radimrehurek.com/gensim/models/keyedvectors.html
            apply_doc (str): Mode to calculate the embeddings vector for the document,
                which could be `"mean"` or `"sum"` of the tokens.
        """
        super().__init__(keyed_vectors=keyed_vectors, apply_doc=apply_doc)
        assert apply_doc in ('sum', 'mean'), 'apply_doc value is not valid, can only be: `"mean"` or `"sum"`.'

        try:
            import numpy as np
            self.np = np
        except ImportError:
            log.error("To use embeddings please install numpy. "
                      "See the docs at https://numpy.org/ for more information.")
            raise

        try:
            import gensim
            assert isinstance(keyed_vectors, gensim.models.KeyedVectors), 'keyed_vectors is not of type `KeyedVectors`'
            self.keyed_vectors = keyed_vectors
            self.apply_doc = apply_doc

        except ImportError:
            log.error("Please install gensim. "
                      "See the docs at https://radimrehurek.com/gensim/ for more information.")
            raise
        self.kwargs['vector_size'] = keyed_vectors.vector_size
        self.kwargs['mapfile_path'] = keyed_vectors.mapfile_path

    @validate(TransformersType.EMBEDDINGS)
    @add_step
    def __call__(self, doc: Document, inplace: bool = False) -> Optional[Document]:
        """Gensim Embedding extraction.

        Args:
            doc (Document): Document to be normalized.
            inplace (bool): if False will return a new doc object,
                            otherwise will change the object passed as parameter.

        Returns: Document
        """
        d = doc if inplace else doc._deepcopy()

        for token in d.tokens:
            try:
                token.embedded = self.keyed_vectors[token.cleaned]
            except KeyError:
                token.embedded = self.np.zeros(self.keyed_vectors.vector_size, dtype=self.np.float32)

        d.embedded = (getattr(self.np, self.apply_doc)([token.embedded for token in d.tokens], axis=0)
                      if len(d.tokens) != 0 else self.np.zeros(self.keyed_vectors.vector_size))

        return None if inplace else d


class TorchTextEmbeddings(BaseTransformer):
    """Torchtext Embeddings extraction.

    Callable arguments:

     Args:
        doc (Document): Document to extract embeddings.
        inplace (bool): if False will return a new doc object,
            otherwise will change the object passed as parameter.

    Returns:
        Document with Torchtext Embedding or None if `inplace=True`.

     Example:

        >>> from torchtext.vocab import CharNGram
        >>> from nlpiper.transformers.embeddings import TorchTextEmbeddings
        >>> from nlpiper.core.document import Document
        >>> from nlpiper.transformers.tokenizers import BasicTokenizer
        >>> doc = Document('Test random stuff.')
        >>> t = BasicTokenizer()
        >>> t(doc, inplace=True)
        >>> model = CharNGram()
        >>> e = TorchTextEmbeddings(model)
        >>> type(e(doc).embedded)
        <class 'numpy.ndarray'>

    """

    def __init__(self, model: Any, apply_doc: str = 'mean', **kwargs):
        """Torchtext Embeddings extraction.

        Args:
            model (Any): Torchtext embedding model. Available models: [`Glove`, `CharNGram`, `FastText`].
                Check further info here: https://pytorch.org/text/stable/vocab.html#pretrained-word-embeddings
            apply_doc (str): Mode to calculate the embeddings vector for the document,
                which could be `"mean"` or `"sum"` of the tokens.

        """
        super().__init__(model=model, apply_doc=apply_doc, **kwargs)
        assert apply_doc in ('sum', 'mean'), 'apply_doc value is not valid, can only be: `"mean"` or `"sum"`.'

        try:
            import numpy as np
            self.np = np
        except ImportError:
            log.error("To use embeddings please install numpy. "
                      "See the docs at https://numpy.org/ for more information.")
            raise

        try:
            import torchtext
            assert issubclass(model.__class__, torchtext.vocab.Vectors), 'vectors is not of type `Vectors`'
            self.model = model
            self.apply_doc = apply_doc

        except ImportError:
            log.error("Please install torchtext. "
                      "See the docs at https://pytorch.org/text/stable/index.html for more information.")
            raise
        self.kwargs['vector_size'] = self.model.dim

    @validate(TransformersType.EMBEDDINGS)
    @add_step
    def __call__(self, doc: Document, inplace: bool = False) -> Optional[Document]:
        """Torchtext Embeddings extraction.

        Args:
            doc (Document): Document to be normalized.
            inplace (bool): if False will return a new doc object,
                otherwise will change the object passed as parameter.

        Returns: Document
        """
        d = doc if inplace else doc._deepcopy()

        for token in d.tokens:
            token.embedded = self.model \
                .get_vecs_by_tokens(token.cleaned) \
                .reshape(-1) \
                .to('cpu') \
                .detach() \
                .numpy()

        d.embedded = (getattr(self.np, self.apply_doc)([token.embedded for token in d.tokens], axis=0)
                      if len(d.tokens) != 0 else self.np.zeros(self.model.dim))

        return None if inplace else d
