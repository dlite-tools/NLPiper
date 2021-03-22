import builtins

import pytest

from nlpiper.transformers.tokenizers import BasicTokenizer, MosesTokenizer
from nlpiper.core.document import Document, Token


@pytest.fixture
def hide_available_pkg(monkeypatch):
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == 'sacremoses':
            raise ModuleNotFoundError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mocked_import)


class TestBasicTokenizer:
    @pytest.mark.parametrize('inputs,results', [
        ('Test to this test', ['Test', 'to', 'this', 'test']),
        ('numbers 123 and symbols "#$%', ['numbers', '123', 'and', 'symbols', '"#$%']),
    ])
    def test_tokenizer(self, inputs, results):
        t = BasicTokenizer()
        input_doc = Document(original=inputs)

        doc = Document(original=inputs, cleaned=inputs, phrases=[inputs])
        doc.tokens = [[Token(original=token) for token in results]]

        assert t(input_doc) == doc


class TestMosesTokenizer:
    @pytest.mark.parametrize('inputs,results', [
        ('Test to this test', ['Test', 'to', 'this', 'test']),
        ('numbers 123 and symbols "#$%', ['numbers', '123', 'and', 'symbols', '&quot;', '#', '$', '%']),
    ])
    def test_tokenizer(self, inputs, results):
        pytest.importorskip('sacremoses')
        t = MosesTokenizer()
        input_doc = Document(original=inputs)

        doc = Document(original=inputs, cleaned=inputs, phrases=[inputs])
        doc.tokens = [[Token(original=token) for token in results]]

        # Given a Document as input
        assert t(input_doc) == doc

        input_doc.phrases = [inputs]
        doc.phrases = [inputs]

        # Given a Document with phrases as input
        assert t(input_doc) == doc

    @pytest.mark.usefixtures('hide_available_pkg')
    def test_if_no_package(self):
        with pytest.raises(ModuleNotFoundError):
            MosesTokenizer()
