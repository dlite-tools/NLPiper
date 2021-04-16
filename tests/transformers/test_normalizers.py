import builtins

import pytest

from nlpiper.transformers.normalizers import (
    CaseTokens,
    RemovePunctuation,
    RemoveStopWords,
    StemmerNLTKSnowball
)
from nlpiper.transformers.tokenizers import BasicTokenizer
from nlpiper.core.document import (
    Document,
    Token
)


@pytest.fixture
def hide_available_pkg(monkeypatch):
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name in ('nltk'):
            raise ModuleNotFoundError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mocked_import)


class TestNormalizersValidations:

    @pytest.mark.parametrize('inputs', ["string", 2])
    def test_with_invalid_input(self, inputs):
        with pytest.raises(TypeError):
            t = CaseTokens()
            t(inputs)

    @pytest.mark.parametrize('inputs', ["test"])
    def test_without_doc_tokens(self, inputs):
        doc = Document("test")

        t = CaseTokens()
        with pytest.raises(RuntimeError):
            t(doc)

    @pytest.mark.usefixtures('hide_available_pkg')
    def test_if_no_package(self):
        with pytest.raises(ModuleNotFoundError):
            RemoveStopWords()

        with pytest.raises(ModuleNotFoundError):
            StemmerNLTKSnowball()


class TestCaseTokens:
    @pytest.mark.parametrize('mode,inputs,results', [
        ('lower', ['TEST'], ['test']),
        ('lower', ['test'], ['test']),
        ('upper', ['test'], ['TEST']),
        ('upper', ['TEST'], ['TEST']),
    ])
    def test_modes(self, mode, inputs, results):

        results_expected = [Token(tk) for tk in inputs]
        for tk, out in zip(results_expected, results):
            tk.cleaned = out

        doc = Document(" ".join(inputs))

        # To apply a normalizer is necessary to have tokens
        t = BasicTokenizer()
        t(doc, inplace=True)

        n = CaseTokens(mode)

        # Inplace False
        out = n(doc)

        assert out.tokens == results_expected
        assert out.steps == [repr(t), repr(n)]
        assert doc.tokens == [Token(token) for token in inputs]
        assert doc.steps == [repr(t)]

        # Inplace True
        out2 = n(doc, True)

        assert doc.tokens == results_expected
        assert doc.steps == [repr(t), repr(n)]
        assert out2 is None

    @pytest.mark.parametrize('mode', [1, 'other'])
    def test_non_existent_mode(self, mode):
        with pytest.raises(ValueError):
            CaseTokens(mode)


class TestRemovePunctuation:
    @pytest.mark.parametrize('inputs,results', [
        (['TEST.%$#"#'], ['TEST']),
        ([r'!"te""!"#$%&()*+,-.s/:;<=>?@[\]^_`{|}~""t'], ['test']),
    ])
    def test_remove_punctuation(self, inputs, results):

        results_expected = [Token(tk) for tk in inputs]
        for tk, out in zip(results_expected, results):
            tk.cleaned = out

        doc = Document(" ".join(inputs))

        # To apply a normalizer is necessary to have tokens
        t = BasicTokenizer()
        t(doc, inplace=True)

        n = RemovePunctuation()
        # Inplace False
        out = n(doc)

        assert out.tokens == results_expected
        assert out.steps == [repr(t), repr(n)]
        assert doc.tokens == [Token(token) for token in inputs]
        assert doc.steps == [repr(t)]

        # Inplace True
        out = n(doc, True)

        assert doc.tokens == results_expected
        assert doc.steps == [repr(t), repr(n)]
        assert out is None


class TestRemoveStopWords:
    @pytest.mark.parametrize('sensitive,inputs,results', [
        (True, ['This', 'is', 'a', 'stop', 'Word'], ['This', '', '', 'stop', 'Word']),
        (False, ['This', 'is', 'a', 'stop', 'Word'], ['', '', '', 'stop', 'Word']),
    ])
    def test_remove_stop_words_w_case_sensitive(self, sensitive, inputs, results):
        pytest.importorskip('nltk')

        results_expected = [Token(tk) for tk in inputs]
        for tk, out in zip(results_expected, results):
            tk.cleaned = out

        doc = Document(" ".join(inputs))

        # To apply a normalizer is necessary to have tokens
        t = BasicTokenizer()
        t(doc, inplace=True)

        n = RemoveStopWords(case_sensitive=sensitive)
        # Inplace False
        out = n(doc)

        assert out.tokens == results_expected
        assert out.steps == [repr(t), repr(n)]
        assert doc.tokens == [Token(token) for token in inputs]
        assert doc.steps == [repr(t)]

        # Inplace True
        out = n(doc, True)

        assert doc.tokens == results_expected
        assert doc.steps == [repr(t), repr(n)]
        assert out is None


class TestStemmerNLTKSnowball:

    @pytest.mark.parametrize('ignore_stopwords,inputs,results', [
        (True, ['This', 'computer', 'is', 'fastest', 'because'], ['this', 'comput', 'is', 'fastest', 'because']),
        (False, ['This', 'computer', 'is', 'fastest', 'because'], ['this', 'comput', 'is', 'fastest', 'becaus'])])
    def test_remove_stop_words_w_case_sensitive(self, ignore_stopwords, inputs, results):
        pytest.importorskip('nltk')

        results_expected = [Token(tk) for tk in inputs]
        for tk, out in zip(results_expected, results):
            tk.cleaned = out

        doc = Document(" ".join(inputs))

        # To apply a normalizer is necessary to have tokens
        t = BasicTokenizer()
        t(doc, inplace=True)

        n = StemmerNLTKSnowball(ignore_stopwords=ignore_stopwords)
        # Inplace False
        out = n(doc)

        assert out.tokens == results_expected
        assert out.steps == [repr(t), repr(n)]
        assert doc.tokens == [Token(token) for token in inputs]
        assert doc.steps == [repr(t)]

        # Inplace True
        out = n(doc, True)

        assert doc.tokens == results_expected
        assert doc.steps == [repr(t), repr(n)]
        assert out is None

