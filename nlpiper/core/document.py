from copy import deepcopy
from typing import Any, List, Optional

from pydantic import BaseModel, validator

from nlpiper.logger import log


def _check_if_embedded_in_numpy_array(v):
    try:
        import numpy as np
        if not isinstance(v, np.ndarray):
            raise TypeError('Embedding value is not a numpy array.')

    except ImportError:
        log.error("To use embeddings please install numpy. "
                  "See the docs at https://numpy.org/ for more information.")
        raise
    return v


class Token(BaseModel):
    original: str
    cleaned: Optional[str] = None
    lemma: Optional[str] = None
    stem: Optional[str] = None
    ner: Optional[str] = None
    embedded: Optional[Any] = None

    def __init__(self, original: str, **data) -> None:
        super().__init__(original=original, cleaned=original, **data)

    @validator('embedded', pre=True)
    def check_if_embedded_in_numpy_array(cls, v):
        return _check_if_embedded_in_numpy_array(v)

    class Config:
        validate_assignment = True


class Document(BaseModel):
    original: str
    cleaned: str
    tokens: Optional[List[Token]] = None
    embedded: Optional[Any] = None
    steps: List[str] = []

    def __init__(self, original: str, **data) -> None:
        super().__init__(original=original, cleaned=original, **data)

    def _deepcopy(self):
        return deepcopy(self)

    @validator('embedded', pre=True)
    def check_if_embedded_in_numpy_array(cls, v):
        return _check_if_embedded_in_numpy_array(v)

    class Config:
        validate_assignment = True
