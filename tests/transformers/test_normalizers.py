import builtins
from copy import deepcopy

import pytest

from nlpiper.transformers.normalizers import CaseTokens, RemovePunctuation, RemoveStopWords
from nlpiper.core.document import Document, Token


@pytest.fixture
def hide_available_pkg(monkeypatch):
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == 'nltk':
            raise ModuleNotFoundError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mocked_import)


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
        doc.cleaned = doc.original
        doc.phrases = phrases
        doc.tokens = [[Token(original=token) for token in phrase] for phrase in inputs]
        input_doc = deepcopy(doc)

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

    @pytest.mark.parametrize('inputs', ["string", 123])
    def test_document_wo_tokens(self, inputs):
        with pytest.raises(TypeError):
            n = CaseTokens()
            n(Document(original=inputs))


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
        doc.cleaned = doc.original
        doc.phrases = phrases
        doc.tokens = [[Token(original=token) for token in phrase] for phrase in inputs]
        input_doc = deepcopy(doc)

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

    @pytest.mark.parametrize('inputs', ["string", 123])
    def test_document_wo_tokens(self, inputs):
        with pytest.raises(TypeError):
            n = RemovePunctuation()
            n(Document(original=inputs))


class TestRemoveStopWords:
    @pytest.mark.parametrize('inputs,results', [
        ([['This', 'is', 'a', 'stop', 'Word']], [['This', '', '', 'stop', 'Word']]),])
    def test_remove_stop_words_w_case_sensitive(self, inputs, results):
        pytest.importorskip('nltk')
        n = RemoveStopWords(case_sensitive=True)

        # Prepare input doc
        phrases = [" ".join(phrase) for phrase in inputs]
        doc = Document(original=" ".join(phrases))
        doc.cleaned = doc.original
        doc.phrases = phrases
        doc.tokens = [[Token(original=token) for token in phrase] for phrase in inputs]
        input_doc = deepcopy(doc)

        # Prepare result doc
        for phrase, phrase_result in zip(doc.tokens, results):
            for token, result in zip(phrase, phrase_result):
                token.cleaned = result

        assert n(input_doc) == doc

    @pytest.mark.parametrize('inputs,results', [
    ([['This', 'is', 'a', 'stop', 'Word']], [['', '', '', 'stop', 'Word']]),])
    def test_remove_stop_words_wo_case_sensitive(self, inputs, results):
        pytest.importorskip('nltk')
        n = RemoveStopWords(case_sensitive=False)

        # Prepare input doc
        phrases = [" ".join(phrase) for phrase in inputs]
        doc = Document(original=" ".join(phrases))
        doc.cleaned = doc.original
        doc.phrases = phrases
        doc.tokens = [[Token(original=token) for token in phrase] for phrase in inputs]
        input_doc = deepcopy(doc)

        # Prepare result doc
        for phrase, phrase_result in zip(doc.tokens, results):
            for token, result in zip(phrase, phrase_result):
                token.cleaned = result

        assert n(input_doc) == doc

    @pytest.mark.parametrize('inputs', ["string", 2])
    def test_with_invalid_document(self, inputs):
        pytest.importorskip('nltk')

        with pytest.raises(TypeError):
            n = RemoveStopWords()
            n(inputs)

    @pytest.mark.parametrize('inputs', ["string", 123])
    def test_document_wo_tokens(self, inputs):
        pytest.importorskip('nltk')

        with pytest.raises(TypeError):
            n = RemoveStopWords()
            n(Document(original=inputs))

    @pytest.mark.usefixtures('hide_available_pkg')
    def test_if_no_package(self):
        with pytest.raises(ModuleNotFoundError):
            RemoveStopWords()
