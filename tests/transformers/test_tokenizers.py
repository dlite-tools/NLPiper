import builtins

import pytest

from nlpiper.transformers.tokenizers import (
    BasicTokenizer,
    MosesTokenizer
)
from nlpiper.core.document import (
    Document,
    Token
)


@pytest.fixture
def hide_available_pkg(monkeypatch):
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name in ('sacremoses'):
            raise ModuleNotFoundError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mocked_import)


class TestTokenizersValidations:

    @pytest.mark.parametrize('inputs', ["string", 2])
    def test_with_invalid_input(self, inputs):
        with pytest.raises(TypeError):
            t = BasicTokenizer()
            t(inputs)

    @pytest.mark.parametrize('inputs', ["test"])
    def test_with_doc_tokens(self, inputs):
        doc = Document("test")
        doc.tokens = list()
        doc.tokens.append(Token("test"))

        t = BasicTokenizer()
        with pytest.raises(RuntimeError):
            t(doc)

    @pytest.mark.usefixtures('hide_available_pkg')
    def test_if_no_package(self):
        with pytest.raises(ModuleNotFoundError):
            MosesTokenizer()


class TestBasicTokenizer:
    @pytest.mark.parametrize('inputs,results', [
        ('Test to this test', ['Test', 'to', 'this', 'test']),
        ('numbers 123 and symbols "#$%', ['numbers', '123', 'and', 'symbols', '"#$%']),
    ])
    def test_tokenizer(self, inputs, results):
        doc = Document(inputs)
        t = BasicTokenizer()

        # Inplace False
        out = t(doc)

        assert out.tokens == [Token(token) for token in results]
        assert out.steps == [repr(t)]
        assert doc.tokens is None
        assert doc.steps == []

        # Inplace True
        out = t(doc, True)

        assert doc.tokens == [Token(token) for token in results]
        assert doc.steps == [repr(t)]
        assert out is None


class TestMosesTokenizer:
    @pytest.mark.parametrize('inputs,results', [
        ('Test to this test', ['Test', 'to', 'this', 'test']),
        ('numbers 123 and symbols "#$%', ['numbers', '123', 'and', 'symbols', '&quot;', '#', '$', '%']),
    ])
    def test_tokenizer(self, inputs, results):
        pytest.importorskip('sacremoses')

        doc = Document(inputs)
        t = MosesTokenizer()

        # Inplace False
        out = t(doc)

        assert out.tokens == [Token(token) for token in results]
        assert out.steps == [repr(t)]
        assert doc.tokens is None
        assert doc.steps == []

        # Inplace True
        out = t(doc, True)

        assert doc.tokens == [Token(token) for token in results]
        assert doc.steps == [repr(t)]
        assert out is None
