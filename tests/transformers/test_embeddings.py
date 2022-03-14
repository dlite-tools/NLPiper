import pytest

from nlpiper.core.document import Document
from nlpiper.transformers.embeddings import (
    GensimEmbeddings,
    TorchTextEmbeddings
)
from nlpiper.transformers.tokenizers import BasicTokenizer
from nlpiper.transformers.normalizers import CaseTokens


def create_gensim_embeddings():
    import numpy as np
    from gensim.models import KeyedVectors

    class Embeddings(KeyedVectors):

        def __init__(self, **kwargs):
            super().__init__(vector_size=25, **kwargs)

        def __getitem__(self, key):
            embeddings = {
                'the': np.array([-1.0167e-02, 2.0194e-02, 2.1473e-01, 1.7289e-01,
                                -4.3659e-01, -1.4687e-01, 1.8429e+00, -1.5753e-01,
                                1.8187e-01, -3.1782e-01, 6.8390e-02, 5.1776e-01,
                                -6.3371e+00, 4.8066e-01, 1.3777e-01, -4.8568e-01,
                                3.9000e-01, -1.9506e-03, -1.0218e-01, 2.1262e-01,
                                -8.6146e-01, 1.7263e-01, 1.8783e-01, -8.4250e-01,
                                -3.1208e-01])
            }
            return embeddings[key]

    return Embeddings()


def create_embeddings_fasttext_glove(tmpdir):
    from torchtext.vocab import Vectors

    p = tmpdir.mkdir('.embeddings').join("vectors.txt")
    p.write("the -0.655379 0.574261 -0.714026 -0.148858 -0.0534275 -1.01101")
    model = Vectors(name=p.basename, cache=p.dirname)
    return model


def create_embeddings_charngram(tmpdir):
    from torchtext.vocab import CharNGram

    p = tmpdir.mkdir('.embeddings').join("charNgram_vectors.txt")
    p.write("3gram-the -0.87907 -0.569777 -0.538168 0.279851 -0.920891 -0.413584")
    CharNGram.name = p.basename
    CharNGram.url = None
    model = CharNGram(cache=p.dirname)
    return model


class TestGensimEmbeddings:
    pytest.importorskip('numpy')
    import numpy as np
    glove_vectors = create_gensim_embeddings()

    @pytest.mark.parametrize('apply_doc', ['sum', 'mean'])
    @pytest.mark.parametrize('document', ['Test random stuff.', ''])
    def test_embedding_token(self, apply_doc, document):
        doc = Document(document)

        # To apply a embedding is necessary to have tokens
        t = BasicTokenizer()
        t(doc, inplace=True)

        e = GensimEmbeddings(self.glove_vectors, apply_doc)

        # Inplace False
        out = e(doc)

        assert out.steps == [repr(t), repr(e)]
        assert out.embedded.shape == (25,)
        assert all([isinstance(token.embedded, self.np.ndarray) for token in out.tokens])
        assert doc.steps == [repr(t)]
        assert doc.embedded is None

        # Inplace True
        out = e(doc, True)

        assert all([isinstance(token.embedded, self.np.ndarray) for token in doc.tokens])
        assert doc.steps == [repr(t), repr(e)]
        assert doc.embedded.shape == (25,)
        assert out is None

    def test_random_apply_doc(self):
        with pytest.raises(AssertionError):
            GensimEmbeddings(self.glove_vectors, 'random')


class TestTorchTextEmbeddings:
    pytest.importorskip('numpy')
    import numpy as np

    @pytest.mark.parametrize('apply_doc', ['sum', 'mean'])
    @pytest.mark.parametrize('document', ['Test random stuff.', ''])
    @pytest.mark.parametrize('emb_model', [create_embeddings_fasttext_glove, create_embeddings_charngram])
    def test_embedings_glove_fasttext(self, apply_doc, document, emb_model, tmpdir):
        doc = Document(document)

        # To apply a embedding is necessary to have tokens
        t = BasicTokenizer()
        t(doc, inplace=True)

        model = emb_model(tmpdir)

        e = TorchTextEmbeddings(model, apply_doc)

        # Inplace False
        out = e(doc)

        assert out.steps == [repr(t), repr(e)]
        assert out.embedded.shape == (6,)
        assert all([isinstance(token.embedded, self.np.ndarray) for token in out.tokens])
        assert doc.steps == [repr(t)]
        assert doc.embedded is None

        # Inplace True
        out = e(doc, True)

        assert all([isinstance(token.embedded, self.np.ndarray) for token in doc.tokens])
        assert doc.steps == [repr(t), repr(e)]
        assert doc.embedded.shape == (6,)
        assert out is None

    def test_random_apply_doc(self, tmpdir):
        model = create_embeddings_fasttext_glove(tmpdir)
        with pytest.raises(AssertionError):
            GensimEmbeddings(model, 'random')


class TestEmbeddings:
    pytest.importorskip('numpy')
    import gensim

    @pytest.mark.parametrize('inputs', ["string", 2])
    def test_with_invalid_input_gensim(self, inputs):
        e = GensimEmbeddings(self.gensim.models.KeyedVectors(1))
        with pytest.raises(TypeError):
            e(inputs)

    @pytest.mark.parametrize('inputs', ["string", 2])
    @pytest.mark.parametrize('emb_model', [create_embeddings_fasttext_glove, create_embeddings_charngram])
    def test_with_invalid_input_torchtext(self, inputs, emb_model, tmpdir):
        model = emb_model(tmpdir)
        t = TorchTextEmbeddings(model)
        with pytest.raises(TypeError):
            t(inputs)

    def test_without_doc_tokens_gensim(self):
        doc = Document("test")

        e = GensimEmbeddings(self.gensim.models.KeyedVectors(1))
        with pytest.raises(RuntimeError):
            e(doc)

    @pytest.mark.parametrize('emb_model', [create_embeddings_fasttext_glove, create_embeddings_charngram])
    def test_without_doc_tokens_torchtext(self, emb_model, tmpdir):
        doc = Document("test")

        model = emb_model(tmpdir)
        e = TorchTextEmbeddings(model)
        with pytest.raises(RuntimeError):
            e(doc)

    @pytest.mark.parametrize('hide_available_pkg', ['numpy', 'gensim'], indirect=['hide_available_pkg'])
    def test_if_no_package_gensim(self, hide_available_pkg):  # noqa: F811
        with pytest.raises(ModuleNotFoundError):
            GensimEmbeddings(1)

    @pytest.mark.parametrize('hide_available_pkg', ['numpy', 'torchtext'], indirect=['hide_available_pkg'])
    @pytest.mark.parametrize('emb_model', [create_embeddings_fasttext_glove, create_embeddings_charngram])
    def test_if_no_package_torchtext(self, emb_model, hide_available_pkg, tmpdir):  # noqa: F811
        model = emb_model(tmpdir)
        with pytest.raises(ModuleNotFoundError):
            TorchTextEmbeddings(model)

    @pytest.mark.parametrize('document', ['Test random stuff.', ''])
    def test_embedding_applied_twice_gensim(self, document):
        doc = Document(document)

        # To apply a embeddings is necessary to have tokens
        t = BasicTokenizer()
        t(doc, inplace=True)

        e = GensimEmbeddings(self.gensim.models.KeyedVectors(1))

        with pytest.raises(RuntimeError):
            e(e(doc))

    @pytest.mark.parametrize('document', ['Test random stuff.', ''])
    @pytest.mark.parametrize('emb_model', [create_embeddings_fasttext_glove, create_embeddings_charngram])
    def test_embedding_applied_twice_torchtext(self, document, emb_model, tmpdir):
        doc = Document(document)

        # To apply a embeddings is necessary to have tokens
        t = BasicTokenizer()
        t(doc, inplace=True)

        model = emb_model(tmpdir)
        e = TorchTextEmbeddings(model)

        with pytest.raises(RuntimeError):
            e(e(doc))

    @pytest.mark.parametrize('document', ['Test random stuff.', ''])
    def test_embedding_applied_then_normalizer_gensim(self, document):
        doc = Document(document)

        # To apply a embeddings is necessary to have tokens
        t = BasicTokenizer()
        n = CaseTokens()
        t(doc, inplace=True)

        e = GensimEmbeddings(self.gensim.models.KeyedVectors(1))

        with pytest.raises(RuntimeError):
            n(e(doc))

    @pytest.mark.parametrize('document', ['Test random stuff.', ''])
    @pytest.mark.parametrize('emb_model', [create_embeddings_fasttext_glove, create_embeddings_charngram])
    def test_embedding_applied_then_normalizer_torchtext(self, document, emb_model, tmpdir):
        doc = Document(document)

        # To apply a embeddings is necessary to have tokens
        t = BasicTokenizer()
        n = CaseTokens()
        t(doc, inplace=True)

        model = emb_model(tmpdir)
        e = TorchTextEmbeddings(model)

        with pytest.raises(RuntimeError):
            n(e(doc))
