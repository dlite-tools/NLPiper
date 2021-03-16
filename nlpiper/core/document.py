from typing import List, Optional

from pydantic import BaseModel


class Token(BaseModel):
    original: str
    processed: Optional[str] = None
    lemma: Optional[str] = None
    stem: Optional[str] = None


class Document(BaseModel):
    text: str
    cleaned: Optional[str] = None
    phrases: Optional[List[str]] = None
    tokens: Optional[List[List[Token]]] = None
