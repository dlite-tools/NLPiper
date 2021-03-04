from collections import OrderedDict

import pytest

from nlpiper.transformers import cleaners, normalizers, tokenizers
from nlpiper.transformers.composition import Compose


class TestCompose:

    @pytest.mark.parametrize('inputs,results', [
        ('T2E1ST.%$#"#', 'TEST'),
        (r'!"t2e""!"#$%&()*+,-.s/:;<=>?@[\]^_`{|}~""t', 'test'),
    ])
    def test_w_cleaners(self, inputs, results):
        crn = cleaners.RemoveNumber()
        nrp = cleaners.RemovePunctuation()

        pipe = Compose([crn, nrp])

        assert pipe(inputs) == results
        assert pipe.log == OrderedDict([("<class 'nlpiper.transformers.cleaners.RemoveNumber'>", {}),
                                        ("<class 'nlpiper.transformers.cleaners.RemovePunctuation'>", {})])

    @pytest.mark.parametrize('inputs,results', [
        (['TEST.%$#"#'], ['test']),
        ([r'!"Te""!"#$%&()*+,-.s/:;<=>?@[\]^_`{|}~""t'], ['test']),
    ])
    def test_w_normalizers(self, inputs, results):
        nct = normalizers.CaseTokens()
        nrp = normalizers.RemovePunctuation()

        pipe = Compose([nct, nrp])

        assert pipe(inputs) == results
        assert pipe.log == OrderedDict([("<class 'nlpiper.transformers.normalizers.CaseTokens'>", {'mode': 'lower'}),
                                        ("<class 'nlpiper.transformers.normalizers.RemovePunctuation'>", {})])

    @pytest.mark.parametrize('inputs,results', [
        ('T2E1ST.%$#"# test', ['test', 'test']),
        (r'!"t2e""!"#$%&()*+,-.s/:;<=>?@[\]^_`{|}~""t', ['test']),
    ])
    def test_w_cleaner_tokenizer_normalizers(self, inputs, results):
        crn = cleaners.RemoveNumber()
        t = tokenizers.BasicTokenizer()
        nct = normalizers.CaseTokens()
        nrp = normalizers.RemovePunctuation()

        pipe = Compose([crn, t, nct, nrp])

        assert pipe(inputs) == results
        assert pipe.log == OrderedDict([("<class 'nlpiper.transformers.cleaners.RemoveNumber'>", {}),
                                        ("<class 'nlpiper.transformers.tokenizers.BasicTokenizer'>", {}),
                                        ("<class 'nlpiper.transformers.normalizers.CaseTokens'>", {'mode': 'lower'}),
                                        ("<class 'nlpiper.transformers.normalizers.RemovePunctuation'>", {})])
