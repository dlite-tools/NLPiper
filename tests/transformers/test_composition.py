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
        crn = cleaners.RemoveNumber()
        nrp = cleaners.RemovePunctuation()
        doc = Document(original=inputs)
        doc.cleaned = results

        pipe = Compose([crn, nrp])

        assert pipe(Document(original=inputs)) == doc

    @pytest.mark.parametrize('inputs,results', [
        ([['TEST.%$#"#']], [['test']]),
        ([[r'!"Te""!"#$%&()*+,-.s/:;<=>?@[\]^_`{|}~""t']], [['test']]),
    ])
    def test_w_normalizers(self, inputs, results):
        nct = normalizers.CaseTokens()
        nrp = normalizers.RemovePunctuation()

        pipe = Compose([nct, nrp])

        phrases = [" ".join(phrase) for phrase in inputs]
        doc = Document(original=" ".join(phrases))
        doc.phrases = phrases
        doc.tokens = [[Token(original=token) for token in phrase] for phrase in inputs]
        input_doc = doc

        for phrase, phrase_result in zip(doc.tokens, results):
            for token, result in zip(phrase, phrase_result):
                token.processed = result

        assert pipe(input_doc) == doc

    @pytest.mark.parametrize('inputs,results', [
        ('T2E1ST.%$#"# test', [['test', 'test']]),
        (r'!"t2e""!"#$%&()*+,-.s/:;<=>?@[\]^_`{|}~""t', [['test']]),
    ])
    def test_w_cleaner_tokenizer_normalizers(self, inputs, results):
        crn = cleaners.RemoveNumber()
        t = tokenizers.BasicTokenizer()
        nct = normalizers.CaseTokens()
        nrp = normalizers.RemovePunctuation()

        pipe = Compose([crn, t, nct, nrp])

        doc = crn(inputs)
        doc.tokens = [[Token(original=token) for token in doc.cleaned.split()]]
        input_doc = doc

        for phrase, phrase_result in zip(doc.tokens, results):
            for token, result in zip(phrase, phrase_result):
                token.processed = result

        assert pipe(input_doc) == doc
