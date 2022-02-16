import pytest

from nlpiper.transformers.normalizers import (
    CaseTokens,
    RemovePunctuation,
    RemoveStopWords,
    VocabularyFilter,
    Stemmer,
    SpellCheck
)
from nlpiper.transformers.tokenizers import BasicTokenizer
from nlpiper.core.document import (
    Document,
    Token
)


from tests.transformers import hide_available_pkg  # noqa: F401


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

    @pytest.mark.parametrize('hide_available_pkg', ['nltk'], indirect=['hide_available_pkg'])
    def test_if_no_package_nltk(self, hide_available_pkg):  # noqa: F811
        with pytest.raises(ModuleNotFoundError):
            RemoveStopWords()

        with pytest.raises(ModuleNotFoundError):
            SpellCheck(max_distance=1)

        with pytest.raises(ModuleNotFoundError):
            Stemmer(version='nltk')

    @pytest.mark.parametrize('hide_available_pkg', ['hunspell'], indirect=['hide_available_pkg'])
    def test_if_no_package_hunspell(self, hide_available_pkg):  # noqa: F811
        with pytest.raises(ModuleNotFoundError):
            SpellCheck(max_distance=None)

        with pytest.raises(ModuleNotFoundError):
            Stemmer(version='hunspell')


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


class TestVocabularyFilter:
    vocabulary = ['this', 'is', 'a', 'token']

    @pytest.mark.parametrize('sensitive,inputs,results', [
        (True, ['This', 'is', 'a', 'Token'], ['', 'is', 'a', '']),
        (False, ['This', 'is', 'a', 'Token'], ['This', 'is', 'a', 'Token']),
    ])
    def test_vocabulary_filter_w_case_sensitive(self, sensitive, inputs, results):
        results_expected = [Token(tk) for tk in inputs]
        for tk, out in zip(results_expected, results):
            tk.cleaned = out

        doc = Document(" ".join(inputs))

        # To apply a normalizer is necessary to have tokens
        t = BasicTokenizer()
        t(doc, inplace=True)

        n = VocabularyFilter(vocabulary=self.vocabulary, case_sensitive=sensitive)
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


class TestSpellCheck:
    @pytest.mark.parametrize('max_distance,inputs,results', [
        (None, ['This', 'isx', 'a', 'stop', 'Word'], ['This', '', 'a', 'stop', 'Word']),
        (1, ['Thisx', 'iszk', 'a', 'stop', 'Word'], ['This', 'iszk', 'a', 'stop', 'Word']),
    ])
    def test_spell_checking(self, max_distance, inputs, results):
        pytest.importorskip('hunspell')
        pytest.importorskip('nltk')

        results_expected = [Token(tk) for tk in inputs]
        for tk, out in zip(results_expected, results):
            tk.cleaned = out

        doc = Document(" ".join(inputs))

        # To apply a normalizer is necessary to have tokens
        t = BasicTokenizer()
        t(doc, inplace=True)

        n = SpellCheck(max_distance=max_distance)
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


class TestStemmer:

    @pytest.mark.parametrize('version,language,inputs,results', [
        ('nltk', 'english', ['This', 'computer', 'is', 'fastest', 'because'],
         ['this', 'comput', 'is', 'fastest', 'becaus']),
        ('hunspell', 'en_GB', ['This', 'computer', 'is', 'fastest', 'because'],
         ['this', 'computer', 'is', 'fast', 'because'])])
    def test_stemmer(self, version, language, inputs, results):
        pytest.importorskip('nltk')
        pytest.importorskip('hunspell')

        results_expected = [Token(tk) for tk in inputs]
        for tk, out in zip(results_expected, results):
            tk.cleaned = out
            tk.stem = out

        doc = Document(" ".join(inputs))

        # To apply a normalizer is necessary to have tokens
        t = BasicTokenizer()
        t(doc, inplace=True)

        n = Stemmer(version=version, language=language)
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

    def test_unavailable_version(self):
        with pytest.raises(ValueError):
            Stemmer(version='random')
