import pytest

from nlpiper.transformers import BaseTransformer


class TestBaseTransformer:

    def test_repr_method(self):
        # Test constructor variations
        base = BaseTransformer()
        assert repr(base) == "BaseTransformer()"

        base = BaseTransformer(1, 'str', 3.0, [1])
        assert repr(base) == "BaseTransformer(1, 'str', 3.0, [1])"

        base = BaseTransformer(a=1, b='str', c=3.0, d=['str', 1])
        assert repr(base) == "BaseTransformer(a=1, b='str', c=3.0, d=['str', 1])"

        base = BaseTransformer(1, 'str', 3.0, a=1, b='str', c=3.0)
        assert repr(base) == "BaseTransformer(1, 'str', 3.0, a=1, b='str', c=3.0)"

        base = BaseTransformer(**{'a': 2, 'b': 4, 'c': 10})
        assert repr(base) == "BaseTransformer(a=2, b=4, c=10)"

    def test_call_raise(self):
        with pytest.raises(NotImplementedError):
            base = BaseTransformer()
            base("test")
