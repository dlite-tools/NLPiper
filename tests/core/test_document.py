import pytest
from pydantic import ValidationError

from nlpiper.core.composition import Compose
from nlpiper.core.document import Document
from nlpiper.transformers.tokenizers import BasicTokenizer


from tests.transformers import hide_available_pkg  # noqa: F401


def create_document():
    pipe = Compose([BasicTokenizer()])
    d = Document('Random Stuff.')
    return pipe(d)


class TestDocument:

    def test_embedding_valid_array(self):
        pytest.importorskip('numpy')
        import numpy as np

        d = create_document()

        d.embedded = np.random.rand(1)
        d.tokens[0].embedded = np.random.rand(1)

    def test_document_embedding_invalid_array(self):
        pytest.importorskip('numpy')

        d = create_document()

        with pytest.raises(ValidationError):
            d.embedded = 1

    def test_token_embedding_invalid_array(self):
        pytest.importorskip('numpy')

        d = create_document()

        with pytest.raises(ValidationError):
            d.tokens[0].embedded = 1

    def test_document_embedding_from_valid_to_invalid_array(self):
        pytest.importorskip('numpy')
        import numpy as np

        d = create_document()

        d.embedded = np.random.rand(1)
        with pytest.raises(ValidationError):
            d.embedded = 1

    def test_token_embedding_from_valid_to_invalid_array(self):
        pytest.importorskip('numpy')
        import numpy as np

        d = create_document()

        d.tokens[0].embedded = np.random.rand(1)
        with pytest.raises(ValidationError):
            d.tokens[0].embedded = 1

    @pytest.mark.parametrize('hide_available_pkg', ['numpy'], indirect=['hide_available_pkg'])
    def test_if_no_package(self, hide_available_pkg):  # noqa: F811
        with pytest.raises(ModuleNotFoundError):
            doc = Document('Test')
            doc.embedded = 1
