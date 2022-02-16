"""Embeddings Module."""

from typing import Any, Optional

from nlpiper.core.document import Document
from nlpiper.transformers import (
    BaseTransformer,
    TransformersType,
    add_step,
    validate
)
from nlpiper.logger import log

__all__ = [
    "GensimEmbeddings"
]


class GensimEmbeddings(BaseTransformer):
    """Gensim Embedding extraction."""

    def __init__(self, keyed_vectors: Any, apply_doc: Optional[str] = None):
        """Stem tokens.

        Gensim Embedding extraction..

        Args:
            keyed_vectors (Any): Gensim model based on keyedVectors,
            see more in: https://radimrehurek.com/gensim/models/keyedvectors.html
            apply_doc (Optional[str]): If different than `None`, then the embeddings vector for the document will
            be calculated, which could be `"mean"` or `"sum"` of the valid tokens.
        """
        super().__init__(keyed_vectors=keyed_vectors, apply_doc=apply_doc)
        assert apply_doc in ('sum', 'mean', None), 'apply_doc value is not valid, can only be:' \
                                                   ' `None`, `"mean"` or `"sum"`.'

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

        if self.apply_doc is not None:
            doc.embedded = getattr(self.np, self.apply_doc)([token.embedded for token in d.tokens], axis=0)

        return None if inplace else d
