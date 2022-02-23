"""Transformers Module."""
from enum import Enum, auto

from nlpiper.core import Document
from nlpiper.logger import log


class BaseTransformer:
    """Base class to all Transformers."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        log.info("[Created] %s", repr(self))

    def __repr__(self) -> str:
        """Create a string representation including init params."""
        params = ', '.join(
            ["%r" % a for a in self.args] +
            ["%s=%r" % (k, v) for k, v in self.kwargs.items()]
        )
        return "%s(%s)" % (self.__class__.__name__, params)

    def __call__(self, doc: Document, inplace: bool = False) -> Document:
        raise NotImplementedError


class TransformersType(Enum):
    CLEANERS = auto()
    TOKENIZERS = auto()
    NORMALIZERS = auto()
    EMBEDDINGS = auto()


# Decorators
def validate(transformer_type: TransformersType):
    """Validate a transformation call.

    Validations:
    - The 'doc' argument must be an instance of Document class.
    - Cleaners can not be called for document with tokens.
    """
    def inner_validate(func):
        def wrapper(*args, **kwargs):
            doc: Document = args[1]
            if not isinstance(doc, Document):
                raise TypeError("Argument doc is not of type Document")

            if transformer_type in (TransformersType.CLEANERS, TransformersType.TOKENIZERS):
                if doc.tokens is not None:
                    raise RuntimeError(
                        f"{transformer_type.name.title()} transformer can not be applied on documents with tokens"
                    )
            elif transformer_type in (TransformersType.NORMALIZERS, TransformersType.EMBEDDINGS):
                if doc.tokens is None:
                    raise RuntimeError(
                        f"{transformer_type.name.title()} transformer can not be applied on documents without tokens"
                    )
                elif doc.embedded is not None:
                    raise RuntimeError(
                        f"{transformer_type.name.title()} transformer can not be applied on documents with embeddings"
                    )
            else:
                raise RuntimeError("TransformerType behavior not implemented")

            return func(*args, **kwargs)
        return wrapper
    return inner_validate


def add_step(func):
    """Register a transformation into the document object."""

    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        if out is None:
            args[1].steps.append(repr(args[0]))
        else:
            out.steps.append(repr(args[0]))

        return out

    return wrapper
