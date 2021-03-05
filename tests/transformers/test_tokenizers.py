import pytest

from nlpiper.transformers.tokenizers import BasicTokenizer
from nlpiper.core.document import Document, Token


class TestRemovePunctuation:
    @pytest.mark.parametrize('inputs,results', [
        ('Test to this test', ['Test', 'to', 'this', 'test']),
        ('numbers 123 and symbols "#$%', ['numbers', '123', 'and', 'symbols', '"#$%']),
    ])
    def test_remove_punctuation(self, inputs, results):
        t = BasicTokenizer()
        input_doc = Document(inputs)

        doc = Document(inputs)
        doc.tokens = [[Token(token) for token in results]]

        assert t(inputs) == doc
        assert t(input_doc) == doc

        input_doc.phrases = [inputs]
        doc.phrases = [inputs]
        assert t(input_doc) == doc
