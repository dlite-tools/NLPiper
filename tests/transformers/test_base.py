from nlpiper.transformers import Base


class TestBaseTransformer:

    def test_repr_method(self):
        # Test constructor variations
        base = Base()
        assert repr(base) == "Base()"

        base = Base(1, 'str', 3.0, [1])
        assert repr(base) == "Base(1, 'str', 3.0, [1])"

        base = Base(a=1, b='str', c=3.0, d=['str', 1])
        assert repr(base) == "Base(a=1, b='str', c=3.0, d=['str', 1])"

        base = Base(1, 'str', 3.0, a=1, b='str', c=3.0)
        assert repr(base) == "Base(1, 'str', 3.0, a=1, b='str', c=3.0)"

        base = Base(**{'a': 2, 'b': 4, 'c': 10})
        assert repr(base) == "Base(a=2, b=4, c=10)"
