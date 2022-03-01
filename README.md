<p align="center">
  <img src="https://raw.githubusercontent.com/tomassosorio/NLPiper/main/docs/imgs/nlpiper.png" />
</p>

[![Test](https://github.com/tomassosorio/NLPiper/actions/workflows/test.yml/badge.svg)](https://github.com/tomassosorio/NLPiper/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/tomassosorio/NLPiper/branch/main/graph/badge.svg?token=PK513BHBVC)](https://codecov.io/gh/tomassosorio/NLPiper)
![Package Version](https://img.shields.io/pypi/v/NLPiper)
![Python Version](https://img.shields.io/pypi/pyversions/NLPiper)

NLPiper is a package that agglomerates different NLP tools and applies their transformations in the target document.

## Goal
Lightweight package integrated with a universe of frameworks to pre-process documents.

---
## Install

The package can be installed using `pip`:

`pip install nlpiper`

For all transforms be available:
`pip install 'nlpiper[all]'`, otherwise, just install the packages needed.

## Usage

### Define a Pipeline:

```python
>>> from nlpiper.core import Compose
>>> from nlpiper.transformers import cleaners, normalizers, tokenizers
>>> pipeline = Compose([
...                    cleaners.CleanNumber(),
...                    tokenizers.BasicTokenizer(),
...                    normalizers.CaseTokens()
... ])
>>> pipeline
Compose([CleanNumber(), BasicTokenizer(), CaseTokens(mode='lower')])
```

### Generate a Document and Document structure:
```python
>>> from nlpiper.core import Document
>>> doc = Document("The following character is a number: 1 and the next one is not a.")
>>> doc
Document(
    original='The following character is a number: 1 and the next one is not a.', 
    cleaned='The following character is a number: 1 and the next one is not a.', 
    tokens=None,
    embedded=None,
    steps=[]
)
```

### Apply Pipeline to a Document:
```python
>>> doc = pipeline(doc)
>>> doc
Document(
    original='The following character is a number: 1 and the next one is not a.', 
    cleaned='The following character is a number:  and the next one is not a.', 
    tokens=[
        Token(original='The', cleaned='the', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='following', cleaned='following', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='character', cleaned='character', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='is', cleaned='is', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='a', cleaned='a', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='number:', cleaned='number:', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='and', cleaned='and', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='the', cleaned='the', lemma=None, stem=None, embedded=None), 
        Token(original='next', cleaned='next', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='one', cleaned='one', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='is', cleaned='is', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='not', cleaned='not', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='a.', cleaned='a.', lemma=None, stem=None, ner=None, embedded=None)
    ],
    embedded=None,
    steps=['CleanNumber()', 'BasicTokenizer()', "CaseTokens(mode='lower')"]
)
```

### Available Transformers
#### Cleaners
Clean document as a whole, e.g. remove HTML, remove accents, remove emails, etc.

- `CleanURL`: remove URL from the text.
- `CleanEmail`: remove email from the text.
- `CleanNumber`: remove numbers from text.
- `CleanPunctuation`: remove punctuation from text.
- `CleanEOF`: remove end of file from text.
- `CleanMarkup`: remove HTML or XML from text.
- `CleanAccents`: remove accents from the text.

#### Tokenizers
Tokenize a document after cleaning is done (Split document into tokens)

- `BasicTokenizer`: Split tokens by spaces in the text.
- `MosesTokenizer`: Split tokens using Moses tokenizer (https://github.com/alvations/sacremoses)
- `StanzaTokenizer`: Split tokens using Stanza tokenizer (https://github.com/stanfordnlp/stanza)

#### Normalizer
Applies on the token level, e.g. remove stop-words, spell-check, etc.

- `CaseTokens`: lower or upper case all tokens.
- `RemovePunctuation`: Remove punctuation from resulting tokens.
- `RemoveStopWords`: Remove stop-words as tokens.
- `VocabularyFilter`: Only allow tokens from a pre-defined vocabulary.
- `Stemmer`: Get the stem from the tokens.
- `SpellCheck`: Spell check the token, if given max distance will calculate the Levenshtein distance from the token with
the suggested word and if lower the token is replaced by the suggestion else will keep the token. If no maximum distance is given if the
word is not correctly spelt then will be replaced by an empty string.

#### Embeddings
Applies on the token level, converting words by embeddings

- `GensimEmbeddings`: Use Gensim word embeddings.

#### Document
`Document` is a dataclass that contains all the information used during text preprocessing.

Document attributes:
- `original`: original text to be processed.
- `cleaned`: original text to be processed when document is initiated and then attribute which `Cleaners` and `Tokenizers` work.
- `tokens`: list of tokens that where obtained using a `Tokenizer`.
- `steps`: list of transforms applied on the document.
- `embedded`: document embedding.

`token`:
- `original`: original token.
- `cleaned`: original token at initiation, then modified according with `Normalizers`.
- `lemma`: token lemma (need to use a normalizer or tokenizer to obtain).
- `stem`: token stem (need to use a normalizer to obtain).
- `ner`: token entity (need to use a normalizer or tokenizer to obtain).
- `embedded`: token embedding.

#### Compose
Compose applies the chosen transformers into a given document.
It restricts the order that the transformers can be applied, first are the Cleaners, then the Tokenizers and lastly
the Normalizers and Embeddings.

It is possible to create a compose using the steps from a processed document:
```python
>>> doc.steps
['CleanNumber()', 'BasicTokenizer()', "CaseTokens(mode='lower')"]
>>> new_pipeline = Compose.create_from_steps(doc.steps)
>>> new_pipeline
Compose([CleanNumber(), BasicTokenizer(), CaseTokens(mode='lower')])
```
It is also possible to rollback the steps applied to a document:
```python
>>> new_doc = Compose.rollback_document(doc, 2)
>>> new_doc
Document(
    original='The following character is a number: 1 and the next one is not a.', 
    cleaned='The following character is a number:  and the next one is not a.', 
    tokens=None,
    embedded=None,
    steps=['CleanNumber()']
)
>>> doc
Document(
    original='The following character is a number: 1 and the next one is not a.', 
    cleaned='The following character is a number:  and the next one is not a.', 
    tokens=[
        Token(original='The', cleaned='the', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='following', cleaned='following', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='character', cleaned='character', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='is', cleaned='is', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='a', cleaned='a', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='number:', cleaned='number:', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='and', cleaned='and', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='the', cleaned='the', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='next', cleaned='next', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='one', cleaned='one', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='is', cleaned='is', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='not', cleaned='not', lemma=None, stem=None, ner=None, embedded=None), 
        Token(original='a.', cleaned='a.', lemma=None, stem=None, ner=None, embedded=None)
    ],
    embedded=None,
    steps=['CleanNumber()', 'BasicTokenizer()', "CaseTokens(mode='lower')"]
)
```

---

## Development Installation

```
git clone https://github.com/tomassosorio/NLPiper.git
cd NLPiper
poetry install
```

---

## Contributions

All contributions, bug reports, bug fixes, documentation improvements,
enhancements and ideas are welcome.

A detailed overview on how to contribute can be found in the
[contributing guide](CONTRIBUTING.md)
on GitHub.

---

## Issues

Go [here](https://github.com/tomassosorio/NLPiper/issues) to submit feature
requests or bugfixes.

---

## License and Credits

`NLPiper` is licensed under the [MIT license](LICENSE) and is written and
maintained by Tomás Osório ([@tomassosorio](https://github.com/tomassosorio)), Daniel Ferrari ([@FerrariDG](https://github.com/FerrariDG)) and Carlos Alves ([@cmalves](https://github.com/cmalves))
