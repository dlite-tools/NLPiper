import builtins

import pytest

from nlpiper.transformers.cleaners import RemovePunctuation, RemoveUrl, RemoveEmail, RemoveNumber, RemoveHTML
from nlpiper.core.document import Document


@pytest.fixture
def hide_available_pkg(monkeypatch):
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == 'bs4':
            raise ModuleNotFoundError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mocked_import)


class TestRemovePunctuation:
    @pytest.mark.parametrize('inputs,results', [
        ('TEST.%$#"#', 'TEST'),
        (r'!"te""!"#$%&()*+,-.s/:;<=>?@[\]^_`{|}~""t', 'test'),
    ])
    def test_remove_punctuation(self, inputs, results):
        r = RemovePunctuation()
        doc = Document(inputs)
        doc.cleaned = results

        assert r(inputs) == doc
        assert r(Document(inputs)) == doc


class TestRemoveUrl:
    @pytest.mark.parametrize('inputs,results', [
        ('TEST', 'TEST'),
        ('test www.web.com', 'test '),
        ('testwww.web.com', 'test'),
        ('test http:\\www.web.com', 'test '),
        ('test https:\\www.web.com', 'test '),
        ('testhttps:\\www.web.com', 'test'),
    ])
    def test_remove_url(self, inputs, results):
        r = RemoveUrl()
        doc = Document(inputs)
        doc.cleaned = results

        assert r(inputs) == doc
        assert r(Document(inputs)) == doc


class TestRemoveEmail:
    @pytest.mark.parametrize('inputs,results', [
        ('TEST', 'TEST'),
        ('test test@test.com', 'test '),
        ('testtest@test.com', ''),
    ])
    def test_remove_email(self, inputs, results):
        r = RemoveEmail()
        doc = Document(inputs)
        doc.cleaned = results

        assert r(inputs) == doc
        assert r(Document(inputs)) == doc


class TestRemoveNumber:
    @pytest.mark.parametrize('inputs,results', [
        ('TEST', 'TEST'),
        ('test 12 test', 'test  test'),
        ('test123test', 'testtest'),
    ])
    def test_remove_number(self, inputs, results):
        r = RemoveNumber()
        doc = Document(inputs)
        doc.cleaned = results

        assert r(inputs) == doc
        assert r(Document(inputs)) == doc


class TestRemoveHTML:
    @pytest.mark.parametrize('inputs,results', [
        ('<html><title>TEST</title>', 'TEST'),
        ('<p class="title"><b>test 12 test</b></p>', 'test 12 test'),
        ('test123test', 'test123test'),
    ])
    def test_remove_html(self, inputs, results):
        pytest.importorskip('bs4')
        r = RemoveHTML()
        doc = Document(inputs)
        doc.cleaned = results

        assert r(inputs) == doc
        assert r(Document(inputs)) == doc
        assert r.log == {"features": "html.parser"}

    @pytest.mark.usefixtures('hide_available_pkg')
    def test_if_no_package(self):
        with pytest.raises(ModuleNotFoundError):
            RemoveHTML()
