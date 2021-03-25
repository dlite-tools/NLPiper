import pytest

from nlpiper.transformers.normalizers import CaseTokens, RemovePunctuation
from nlpiper.core.document import Document, Token


class TestCaseTokens:
    @pytest.mark.parametrize('mode,inputs,results', [
        ('lower', [['TEST']], [['test']]),
        ('upper', [['test']], [['TEST']]),
    ])
    def test_modes(self, mode, inputs, results):
        n = CaseTokens(mode)

        # Prepare input doc
        phrases = [" ".join(phrase) for phrase in inputs]
        doc = Document(original=" ".join(phrases))
        doc.phrases = phrases
        doc.tokens = [[Token(original=token) for token in phrase] for phrase in inputs]
        input_doc = doc

        # Prepare result doc
        for phrase, phrase_result in zip(doc.tokens, results):
            for token, result in zip(phrase, phrase_result):
                token.cleaned = result

        assert n(input_doc) == doc

    @pytest.mark.parametrize('mode', [1, 'other'])
    def test_non_existent_mode(self, mode):
        with pytest.raises(AssertionError):
            CaseTokens(mode)

    @pytest.mark.parametrize('inputs', ["string", 2])
    def test_with_invalid_document(self, inputs):
        with pytest.raises(TypeError):
            n = CaseTokens()
            n(inputs)


class TestRemovePunctuation:
    @pytest.mark.parametrize('inputs,results', [
        ([['TEST.%$#"#']], [['TEST']]),
        ([[r'!"te""!"#$%&()*+,-.s/:;<=>?@[\]^_`{|}~""t']], [['test']]),
    ])
    def test_remove_punctuation(self, inputs, results):
        n = RemovePunctuation()

        # Prepare input doc
        phrases = [" ".join(phrase) for phrase in inputs]
        doc = Document(original=" ".join(phrases))
        doc.phrases = phrases
        doc.tokens = [[Token(original=token) for token in phrase] for phrase in inputs]
        input_doc = doc

        # Prepare result doc
        for phrase, phrase_result in zip(doc.tokens, results):
            for token, result in zip(phrase, phrase_result):
                token.cleaned = result

        assert n(input_doc) == doc

    @pytest.mark.parametrize('inputs', ["string", 2])
    def test_with_invalid_document(self, inputs):
        with pytest.raises(TypeError):
            n = RemovePunctuation()
            n(inputs)
