import pytest

from nlpiper.transformers.cleaners import RemovePunctuation, RemoveUrl, RemoveEmail, RemoveNumber


class TestRemovePunctuation:
    @pytest.mark.parametrize('inputs,results', [
        ('TEST.%$#"#', 'TEST'),
        (r'!"te""!"#$%&()*+,-.s/:;<=>?@[\]^_`{|}~""t', 'test'),
    ])
    def test_remove_punctuation(self, inputs, results):
        r = RemovePunctuation()
        assert r(inputs) == results


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
        assert r(inputs) == results


class TestRemoveEmail:
    @pytest.mark.parametrize('inputs,results', [
        ('TEST', 'TEST'),
        ('test test@test.com', 'test '),
        ('testtest@test.com', ''),
    ])
    def test_remove_email(self, inputs, results):
        r = RemoveEmail()
        assert r(inputs) == results


class TestNumbers:
    @pytest.mark.parametrize('inputs,results', [
        ('TEST', 'TEST'),
        ('test 12 test', 'test  test'),
        ('test123test', 'testtest'),
    ])
    def test_remove_number(self, inputs, results):
        r = RemoveNumber()
        assert r(inputs) == results
