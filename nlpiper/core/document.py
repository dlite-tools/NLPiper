from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Token:
    original: str
    processed: Optional[str] = None
    lemma: Optional[str] = None
    stem: Optional[str] = None


@dataclass
class Document:
    text: str
    cleaned: Optional[str] = None
    phrases: Optional[List[str]] = None
    tokens: Optional[List[List[Token]]] = None
