from copy import deepcopy

import pytest

from nlpiper.transformers import cleaners, normalizers, tokenizers
from nlpiper.core.composition import Compose
from nlpiper.core.document import Document, Token


class TestCompose:

    @pytest.mark.parametrize('inputs,results', [
        ('T2E1ST.%$#"#', 'TEST'),
        (r'!"t2e""!"#$%&()*+,-.s/:;<=>?@[\]^_`{|}~""t', 'test'),
    ])
    def test_w_cleaners(self, inputs, results):
        crn = cleaners.CleanNumber()
        nrp = cleaners.CleanPunctuation()
        doc = Document(inputs)

        pipe = Compose([crn, nrp])

        out = pipe(doc)

        assert out.cleaned == results
        assert out.steps == [repr(crn), repr(nrp)]
        assert doc.cleaned == inputs
        assert doc.steps == []

    @pytest.mark.parametrize('inputs,results', [
        (['TEST.%$#"#'], ['test']),
        ([r'!"Te""!"#$%&()*+,-.s/:;<=>?@[\]^_`{|}~""t'], ['test']),
    ])
    def test_w_normalizers(self, inputs, results):
        results_expected = [Token(tk) for tk in inputs]
        for tk, out in zip(results_expected, results):
            tk.cleaned = out

        doc = Document(" ".join(inputs))

        t = tokenizers.BasicTokenizer()
        nct = normalizers.CaseTokens()
        nrp = normalizers.RemovePunctuation()

        pipe = Compose([t, nct, nrp])

        out = pipe(doc)

        assert out.tokens == results_expected
        assert out.steps == [repr(t), repr(nct), repr(nrp)]
        assert doc.tokens is None
        assert doc.steps == []

    @pytest.mark.parametrize('inputs,results', [
        (['T2E1ST.%$#"#', 'test'], [('TEST.%$#"#', 'test'), ('test', 'test')]),
        ([r'!"t2e""!"#$%&()*+,-.s/:;<=>?@[\]^_`{|}~""t'], [(r'!"te""!"#$%&()*+,-.s/:;<=>?@[\]^_`{|}~""t', 'test')]),
    ])
    def test_w_cleaner_tokenizer_normalizers(self, inputs, results):
        results_expected = [Token(tk_original) for tk_original, tk_cleaned in results]
        for tk, (tk_original, tk_cleaned) in zip(results_expected, results):
            tk.cleaned = tk_cleaned

        doc = Document(" ".join(inputs))

        crn = cleaners.CleanNumber()
        t = tokenizers.BasicTokenizer()
        nct = normalizers.CaseTokens()
        nrp = normalizers.RemovePunctuation()

        pipe = Compose([crn, t, nct, nrp])

        out = pipe(doc)

        assert out.tokens == results_expected
        assert out.steps == [repr(crn), repr(t), repr(nct), repr(nrp)]
        assert doc.tokens is None
        assert doc.steps == []

    def test_create_compose_from_steps(self):

        pipe = Compose([
            cleaners.CleanNumber(),
            tokenizers.BasicTokenizer(),
            normalizers.CaseTokens(),
            normalizers.RemovePunctuation()
        ])

        doc = Document("basic test document 1.")
        out = pipe(doc)

        new_pipe = Compose.create_from_steps(out.steps)

        assert repr(pipe) == repr(new_pipe)

    def test_create_compose_from_steps_with_wrong_def(self):

        pipe = Compose([
            cleaners.CleanNumber(),
            tokenizers.BasicTokenizer(),
            normalizers.CaseTokens(),
            normalizers.RemovePunctuation()
        ])

        doc = Document("basic test document 1.")
        out = pipe(doc)
        out.steps.append('NotTransformer()')

        with pytest.raises(NameError):
            new_pipe = Compose.create_from_steps(out.steps)
