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
