import pytest

from nlpiper.transformers.normalizers import CaseTokens, RemovePunctuation
from nlpiper.core.document import Document, Token


class TestCaseTokens:
    @pytest.mark.parametrize('mode,inputs,results', [
        ('lower', [['TEST']], [['test']]),
        ('upper', [['test']], [['TEST']]),
    ])
    def test_modes(self, mode, inputs, results):
        c = CaseTokens(mode)

        phrases = [" ".join(phrase) for phrase in inputs]
        doc = Document(original=" ".join(phrases))
        doc.phrases = phrases
        doc.tokens = [[Token(original=token) for token in phrase] for phrase in inputs]
        input_doc = doc

        for phrase, phrase_result in zip(doc.tokens, results):
            for token, result in zip(phrase, phrase_result):
                token.processed = result

        assert c(inputs) == doc
        assert c(input_doc) == doc
        assert c.log == {"mode": mode}

    @pytest.mark.parametrize('mode', [1, 'other'])
    def test_non_existent_mode(self, mode):
        with pytest.raises(AssertionError):
            CaseTokens(mode)


class TestRemovePunctuation:
    @pytest.mark.parametrize('inputs,results', [
        ([['TEST.%$#"#']], [['TEST']]),
        ([[r'!"te""!"#$%&()*+,-.s/:;<=>?@[\]^_`{|}~""t']], [['test']]),
    ])
    def test_remove_punctuation(self, inputs, results):
        r = RemovePunctuation()

        phrases = [" ".join(phrase) for phrase in inputs]
        doc = Document(original=" ".join(phrases))
        doc.phrases = phrases
        doc.tokens = [[Token(original=token) for token in phrase] for phrase in inputs]
        input_doc = doc

        for phrase, phrase_result in zip(doc.tokens, results):
            for token, result in zip(phrase, phrase_result):
                token.processed = result

        assert r(inputs) == doc
        assert r(input_doc) == doc
