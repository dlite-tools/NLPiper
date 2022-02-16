import pytest

from nlpiper.core.document import Document
from nlpiper.transformers.embeddings import GensimEmbeddings
from nlpiper.transformers.tokenizers import BasicTokenizer


from tests.transformers import hide_available_pkg  # noqa: F401


class TestGensimEmbeddings:
    pytest.importorskip('gensim')
    pytest.importorskip('numpy')
    import gensim.downloader
    import numpy as np
    glove_vectors = gensim.downloader.load('glove-twitter-25')

    def test_embedding_token(self):
        doc = Document('Test random stuff.')

        # To apply a embeddings is necessary to have tokens
        t = BasicTokenizer()
        t(doc, inplace=True)

        e = GensimEmbeddings(self.glove_vectors)

        # Inplace False
        out = e(doc)

        assert out.steps == [repr(t), repr(e)]
        assert all([isinstance(token.embedded, self.np.ndarray) for token in out.tokens])
        assert all([token.embedded is None for token in doc.tokens])
        assert doc.steps == [repr(t)]
        assert doc.embedded is None

        # Inplace True
        out = e(doc, True)

        assert all([isinstance(token.embedded, self.np.ndarray) for token in doc.tokens])
        assert doc.steps == [repr(t), repr(e)]
        assert doc.embedded is None
        assert out is None

    @pytest.mark.parametrize('apply_doc', ['sum', 'mean'])
    def test_embedding_token_document(self, apply_doc):
        doc = Document('Test random stuff.')

        # To apply a embeddings is necessary to have tokens
        t = BasicTokenizer()
        t(doc, inplace=True)

        e = GensimEmbeddings(self.glove_vectors, apply_doc)

        # Inplace False
        out = e(doc)

        assert out.steps == [repr(t), repr(e)]
        assert all([isinstance(token.embedded, self.np.ndarray) for token in out.tokens])
        assert all([token.embedded is None for token in doc.tokens])
        assert doc.steps == [repr(t)]
        assert doc.embedded.shape == (25,)

        # Inplace True
        out = e(doc, True)

        assert all([isinstance(token.embedded, self.np.ndarray) for token in doc.tokens])
        assert doc.steps == [repr(t), repr(e)]
        assert doc.embedded.shape == (25,)
        assert out is None

    def test_random_apply_doc(self):
        with pytest.raises(AssertionError):
            GensimEmbeddings(self.glove_vectors, 'random')


class TestEmbeddings:
    pytest.importorskip('hunspell')
    pytest.importorskip('numpy')
    import gensim

    @pytest.mark.parametrize('inputs', ["string", 2])
    def test_with_invalid_input(self, inputs):
        with pytest.raises(TypeError):
            t = GensimEmbeddings(self.gensim.models.KeyedVectors(1))
            t(inputs)

    @pytest.mark.parametrize('inputs', ["test"])
    def test_without_doc_tokens(self, inputs):
        doc = Document("test")

        t = GensimEmbeddings(self.gensim.models.KeyedVectors(1))
        with pytest.raises(RuntimeError):
            t(doc)

    @pytest.mark.parametrize('hide_available_pkg', ['numpy', 'gensim'], indirect=['hide_available_pkg'])
    def test_if_no_package(self, hide_available_pkg):  # noqa: F811
        with pytest.raises(ModuleNotFoundError):
            GensimEmbeddings(1)
