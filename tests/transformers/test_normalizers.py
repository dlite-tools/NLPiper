import pytest

from nlpiper.transformers.normalizers import CaseTokens, RemovePunctuation


class TestCaseTokens:
    @pytest.mark.parametrize('mode,inputs,results', [
        ('lower', ['TEST'], ['test']),
        ('upper', ['test'], ['TEST']),
    ])
    def test_modes(self, mode, inputs, results):
        c = CaseTokens(mode)
        assert c(inputs) == results

    @pytest.mark.parametrize('mode', [1, 'other'])
    def test_non_existent_mode(self, mode):
        with pytest.raises(AssertionError):
            CaseTokens(mode)


class TestRemovePunctuation:
    @pytest.mark.parametrize('inputs,results', [
        (['TEST.%$#"#'], ['TEST']),
        ([r'!"te""!"#$%&()*+,-.s/:;<=>?@[\]^_`{|}~""t'], ['test']),
    ])
    def test_remove_punctuation(self, inputs, results):
        r = RemovePunctuation()
        assert r(inputs) == results
