import pytest

from nlpiper.transformers.tokenizers import (
    BasicTokenizer,
    MosesTokenizer,
    StanzaTokenizer,
    SpacyTokenizer,
)
from nlpiper.core.document import (
    Document,
    Token
)


class TestTokenizersValidations:

    @pytest.mark.parametrize('inputs', ["string", 2])
    def test_with_invalid_input(self, inputs):
        t = BasicTokenizer()

        with pytest.raises(TypeError):
            t(inputs)

    @pytest.mark.parametrize('inputs', ["test"])
    def test_with_doc_tokens(self, inputs):
        doc = Document("test")
        doc.tokens = list()
        doc.tokens.append(Token("test"))

        t = BasicTokenizer()

        with pytest.raises(RuntimeError):
            t(doc)

    @pytest.mark.parametrize('hide_available_pkg,package', [('sacremoses', MosesTokenizer),
                                                            ('stanza', StanzaTokenizer),
                                                            ('spacy', SpacyTokenizer), ],
                             indirect=['hide_available_pkg'])
    def test_if_no_package(self, hide_available_pkg, package):  # noqa: F811
        with pytest.raises(ModuleNotFoundError):
            package()


class TestBasicTokenizer:
    @pytest.mark.parametrize('inputs,results', [
        ('Test to this test', ['Test', 'to', 'this', 'test']),
        ('numbers 123 and symbols "#$%', ['numbers', '123', 'and', 'symbols', '"#$%']),
    ])
    def test_tokenizer(self, inputs, results):
        doc = Document(inputs)
        t = BasicTokenizer()

        # Inplace False
        out = t(doc)

        assert out.tokens == [Token(token) for token in results]
        assert out.steps == [repr(t)]
        assert doc.tokens is None
        assert doc.steps == []

        # Inplace True
        out = t(doc, True)

        assert doc.tokens == [Token(token) for token in results]
        assert doc.steps == [repr(t)]
        assert out is None


class TestMosesTokenizer:
    @pytest.mark.parametrize('inputs,results', [
        ('Test to this test', ['Test', 'to', 'this', 'test']),
        ('numbers 123 and symbols "#$%', ['numbers', '123', 'and', 'symbols', '&quot;', '#', '$', '%']),
    ])
    def test_tokenizer(self, inputs, results):
        pytest.importorskip('sacremoses')

        doc = Document(inputs)
        t = MosesTokenizer()

        # Inplace False
        out = t(doc)

        assert out.tokens == [Token(token) for token in results]
        assert out.steps == [repr(t)]
        assert doc.tokens is None
        assert doc.steps == []

        # Inplace True
        out = t(doc, True)

        assert doc.tokens == [Token(token) for token in results]
        assert doc.steps == [repr(t)]
        assert out is None


class TestStanzaTokenizer:
    @pytest.mark.parametrize('inputs,results', [
        ('Test to this test', ['Test', 'to', 'this', 'test']),
        ('numbers 123 and symbols "#$%', ['numbers', '123', 'and', 'symbols', '"', '#', '$', '%']),
    ])
    def test_tokenizer(self, inputs, results):
        pytest.importorskip('stanza')

        doc = Document(inputs)
        t = StanzaTokenizer()

        # Inplace False
        out = t(doc)

        assert out.tokens == [Token(token) for token in results]
        assert out.steps == [repr(t)]
        assert doc.tokens is None
        assert doc.steps == []

        # Inplace True
        out = t(doc, True)

        assert doc.tokens == [Token(token) for token in results]
        assert doc.steps == [repr(t)]
        assert out is None

    @pytest.mark.parametrize('processors,document,tokenization_result,lemma_result,ner_result', [
        ('tokenize,mwt,pos,lemma,ner',
         'Nous, Tomás, Carlos un Daniel, avons atteint la fin du sentier.',
         ['Nous', ',', 'Tomás', ',', 'Carlos', 'un', 'Daniel', ',', 'avons', 'atteint', 'la',
          'fin', 'du', 'du', 'sentier', '.'],
         ['il', ',', 'Tomás', ',', 'Carlos', 'un', 'Daniel', ',', 'avoir', 'atteindre', 'le', 'fin',
          'de', 'le', 'sentier', '.'],
         ['O', 'O', 'S-PER', 'O', 'S-PER', 'O', 'S-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
         ),

        ('tokenize,mwt,pos,lemma',
         'Nous, Tomás, Carlos un Daniel, avons atteint la fin du sentier.',
         ['Nous', ',', 'Tomás', ',', 'Carlos', 'un', 'Daniel', ',', 'avons', 'atteint', 'la',
          'fin', 'du', 'du', 'sentier', '.'],
         ['il', ',', 'Tomás', ',', 'Carlos', 'un', 'Daniel', ',', 'avoir', 'atteindre', 'le', 'fin',
          'de', 'le', 'sentier', '.'],
         [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
         ),

        ('tokenize,ner',
         'Nous, Tomás Carlos un Daniel, avons atteint la fin du sentier.',
         ['Nous', ',', 'Tomás', 'Carlos', 'un', 'Daniel', ',', 'avons', 'atteint', 'la', 'fin', 'du', 'du',
          'sentier', '.'],
         [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
         ['O', 'O', 'B-PER', 'E-PER', 'O', 'S-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
         )
    ])
    def test_tokenizer_w_processors_mtw_pos_lemma_ner(self, processors, document, tokenization_result, lemma_result,
                                                      ner_result):
        pytest.importorskip('stanza')
        tokens_result = [Token(original=token, lemma=lemma, ner=ner)
                         for token, lemma, ner in zip(tokenization_result, lemma_result, ner_result)]

        doc = Document(document)
        t = StanzaTokenizer(language='fr', processors=processors)

        # Inplace False
        out = t(doc)

        assert out.tokens == tokens_result
        assert out.steps == [repr(t)]
        assert doc.tokens is None
        assert doc.steps == []

        # Inplace True
        out = t(doc, True)

        assert doc.tokens == tokens_result
        assert doc.steps == [repr(t)]
        assert out is None

    def test_wrong_processor(self):
        pytest.importorskip('stanza')
        with pytest.raises(AssertionError):
            StanzaTokenizer(processors='ner')


class TestSpacyTokenizer:

    @pytest.mark.parametrize('inputs,lemma,ner,ner_iob,tag,results,kwargs', [
        ('Test to this test',
         ['test', 'to', 'this', 'test'],  # lemma
         ['', '', '', ''],  # ner
         ['O', 'O', 'O', 'O'],  # ner_iob
         ['VB', 'IN', 'DT', 'NN'],  # tag
         ['Test', 'to', 'this', 'test'],  # results
         {}
         ),

        ('Test to this test',
         ['', '', '', ''],  # lemma
         ['', '', '', ''],  # ner
         ['O', 'O', 'O', 'O'],  # ner_iob
         ['VB', 'IN', 'DT', 'NN'],  # tag
         ['Test', 'to', 'this', 'test'],  # results
         {'disable': ['lemmatizer']}
         ),

        ('numbers 123 and symbols "#$%',
         ['number', '123', 'and', 'symbol', '"', '#', '$', '%'],  # lemma
         ['', 'CARDINAL', '', '', '', 'MONEY', 'MONEY', 'MONEY'],  # ner
         ['O', 'B', 'O', 'O', 'O', 'B', 'I', 'I'],  # ner_iob
         ['NNS', 'CD', 'CC', 'NNS', '``', '$', '$', 'NN'],  # tag
         ['numbers', '123', 'and', 'symbols', '"', '#', '$', '%'],  # results
         {}
         ),
    ])
    def test_tokenizer(self, inputs, lemma, ner, ner_iob, tag, results, kwargs):
        pytest.importorskip('spacy')
        import spacy
        spacy.cli.download('en_core_web_sm')

        doc = Document(inputs)
        t = SpacyTokenizer(name='en_core_web_sm', **kwargs)

        # Inplace False
        out = t(doc)

        assert out.tokens == [Token(original=token, lemma=lem, ner=n, ner_iob=n_iob, tag=t)
                              for lem, n, n_iob, t, token in zip(lemma, ner, ner_iob, tag, results)]
        assert out.steps == [repr(t)]
        assert doc.tokens is None
        assert doc.steps == []

        # Inplace True
        out = t(doc, True)

        assert doc.tokens == [Token(original=token, lemma=lem, ner=n, ner_iob=n_iob, tag=t)
                              for lem, n, n_iob, t, token in zip(lemma, ner, ner_iob, tag, results)]
        assert doc.steps == [repr(t)]
        assert out is None

    def test_wrong_processor(self):
        pytest.importorskip('spacy')

        with pytest.raises(IOError):
            SpacyTokenizer(name='random_model')
