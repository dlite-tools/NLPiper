from typing import List, Optional
from pydantic import BaseModel


class Token(BaseModel):
    original: str
    cleaned: Optional[str] = None
    lemma: Optional[str] = None
    stem: Optional[str] = None

    def __init__(self, original: str, **data) -> None:
        super().__init__(original=original, cleaned=original, **data)


class Document(BaseModel):
    original: str
    cleaned: str
    tokens: Optional[List[Token]] = None
    steps: List[str] = []

    def __init__(self, original: str, **data) -> None:
        super().__init__(original=original, cleaned=original, **data)


# Decorators
def validate_doc(func):
    """Validate function first argument as a Document instance."""

    def wrapper(*args, **kwargs):
        if not isinstance(args[1], Document):
            raise TypeError("Argument doc is not of type Document")
        out = func(*args, **kwargs)
        return out

    return wrapper


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
