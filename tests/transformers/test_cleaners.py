import pytest

from nlpiper.transformers.cleaners import (
    CleanAccents,
    CleanEmail,
    CleanEOF,
    CleanMarkup,
    CleanNumber,
    CleanPunctuation,
    CleanURL,
)
from nlpiper.core.document import (
    Document,
    Token
)


class TestCleanersValidations:

    @pytest.mark.parametrize('inputs', ["string", 2])
    def test_with_invalid_input(self, inputs):
        with pytest.raises(TypeError):
            c = CleanEOF()
            c(inputs)

    @pytest.mark.parametrize('inputs', ["test"])
    def test_with_doc_tokens(self, inputs):
        doc = Document("test")
        doc.tokens = list()
        doc.tokens.append(Token("test"))

        c = CleanEOF()
        with pytest.raises(RuntimeError):
            c(doc)

    @pytest.mark.parametrize('hide_available_pkg', ['bs4'], indirect=['hide_available_pkg'])
    def test_if_no_package(self, hide_available_pkg):  # noqa: F811
        with pytest.raises(ModuleNotFoundError):
            CleanMarkup()


class TestCleanURL:
    @pytest.mark.parametrize('inputs,results', [
        ('TEST', 'TEST'),
        ('test www.web.com', 'test '),
        ('testwww.web.com', 'test'),
        ('test http:\\www.web.com', 'test '),
        ('test https:\\www.web.com', 'test '),
        ('testhttps:\\www.web.com', 'test'),
    ])
    def test_clean_url(self, inputs, results):
        doc = Document(inputs)

        # Inplace False
        c = CleanURL()
        out = c(doc)

        assert out.cleaned == results
        assert out.steps == [repr(c)]
        assert doc.cleaned == inputs
        assert doc.steps == []

        # Inplace True
        out = c(doc, True)

        assert doc.cleaned == results
        assert doc.steps == [repr(c)]
        assert out is None


class TestCleanEmail:
    @pytest.mark.parametrize('inputs,results', [
        ('TEST', 'TEST'),
        ('test test@test.com', 'test '),
        ('testtest@test.com', ''),
        ('testtest@test.org', ''),
    ])
    def test_clean_email(self, inputs, results):
        doc = Document(inputs)
        c = CleanEmail()

        # Inplace False
        out = c(doc)

        assert out.cleaned == results
        assert out.steps == [repr(c)]
        assert doc.cleaned == inputs
        assert doc.steps == []

        # Inplace True
        out = c(doc, True)

        assert doc.cleaned == results
        assert doc.steps == [repr(c)]
        assert out is None


class TestCleanNumber:
    @pytest.mark.parametrize('inputs,results', [
        ('TEST', 'TEST'),
        ('test 12 test', 'test  test'),
        ('test123test', 'testtest'),
    ])
    def test_clean_number(self, inputs, results):
        doc = Document(inputs)
        c = CleanNumber()

        # Inplace False
        out = c(doc)

        assert out.cleaned == results
        assert out.steps == [repr(c)]
        assert doc.cleaned == inputs
        assert doc.steps == []

        # Inplace True
        out = c(doc, True)

        assert doc.cleaned == results
        assert doc.steps == [repr(c)]
        assert out is None


class TestCleanPunctuation:
    @pytest.mark.parametrize('inputs,results', [
        ('TEST.%$#"#', 'TEST'),
        (r'!"te""!"#$%&()*+,-.s/:;<=>?@[\]^_`{|}~""t', 'test'),
    ])
    def test_clean_punctuation(self, inputs, results):
        doc = Document(inputs)
        c = CleanPunctuation()

        # Inplace False
        out = c(doc)

        assert out.cleaned == results
        assert out.steps == [repr(c)]
        assert doc.cleaned == inputs
        assert doc.steps == []

        # Inplace True
        out = c(doc, True)

        assert doc.cleaned == results
        assert doc.steps == [repr(c)]
        assert out is None


class TestCleanMarkup:
    @pytest.mark.parametrize('inputs,results', [
        ('<html><title>TEST</title>', 'TEST'),
        ('<p class="title"><b>test 12 test</b></p>', 'test 12 test'),
        ('<?xml version="1.0" encoding="UTF-8"?><note><body>test123test</body></note>', 'test123test'),
        ('<html><title>TEST<br><br>TEST</title>', 'TEST TEST'),
    ])
    def test_clean_markup(self, inputs, results):
        pytest.importorskip('bs4')
        doc = Document(inputs)
        c = CleanMarkup()

        # Inplace False
        out = c(doc)

        assert out.cleaned == results
        assert out.steps == [repr(c)]
        assert doc.cleaned == inputs
        assert doc.steps == []

        # Inplace True
        out = c(doc, True)

        assert doc.cleaned == results
        assert doc.steps == [repr(c)]
        assert out is None


class TestCleanAccents:
    @pytest.mark.parametrize('mode,inputs,results', [
        ('unicode', 'àáâãäåçèéêë', 'aaaaaaceeee'),
        ('ascii', 'àáâãäåçèéêë', 'aaaaaaceeee'),
        ('unicode', 'ìíîïñòóôõöùúûüý', 'iiiinooooouuuuy'),
        ('ascii', 'ìíîïñòóôõöùúûüý', 'iiiinooooouuuuy'),
        ('unicode', 'this is à test', 'this is a test'),
        ('unicode', 'this is a test', 'this is a test'),
        ('ascii', 'this is à test', 'this is a test'),
        ('unicode', '\u0625', '\u0627'),
        ('ascii', '\u0625', ''),
        ('unicode', 'o\u0308', 'o'),
        ('ascii', 'o\u0308', 'o'),
        ('unicode', '\u0300\u0301\u0302\u0303', ''),
        ('ascii', '\u0300\u0301\u0302\u0303', ''),
        ('unicode', 'o\u0308\u0304', 'o'),
        ('ascii', 'o\u0308\u0304', 'o'),
    ])
    def test_clean_accents(self, mode, inputs, results):
        doc = Document(inputs)
        c = CleanAccents(mode=mode)

        # Inplace False
        out = c(doc)

        assert out.cleaned == results
        assert out.steps == [repr(c)]
        assert doc.cleaned == inputs
        assert doc.steps == []

        # Inplace True
        out = c(doc, True)

        assert doc.cleaned == results
        assert doc.steps == [repr(c)]
        assert out is None

    @pytest.mark.parametrize('mode', ["random", 2])
    def test_with_invalid_mode(self, mode):
        with pytest.raises(ValueError):
            CleanAccents(mode=mode)


class TestCleanEOF:
    @pytest.mark.parametrize('inputs,results', [
        ('', ''),
        ('a basic phrase', 'a basic phrase'),
        ('line\nline', 'line line'),
        ('line.\nline', 'line. line')
    ])
    def test_clean_eof(self, inputs, results):
        doc = Document(inputs)
        c = CleanEOF()

        # Inplace False
        out = c(doc)

        assert out.cleaned == results
        assert out.steps == [repr(c)]
        assert doc.cleaned == inputs
        assert doc.steps == []

        # Inplace True
        out = c(doc, True)

        assert doc.cleaned == results
        assert doc.steps == [repr(c)]
        assert out is None
