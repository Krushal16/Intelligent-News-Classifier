# src/preprocess.py
import re
from typing import Iterable, List

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def apply_cleaning(texts: Iterable[str]) -> List[str]:
    return [clean_text(t) for t in texts]

