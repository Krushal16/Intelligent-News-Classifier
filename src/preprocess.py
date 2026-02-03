# src/preprocess.py
import re
from typing import Iterable

def clean_text(text: str) -> str:
    """
    Minimal cleaning: lowercase and strip extra whitespace.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def apply_cleaning(texts: Iterable[str]):
    """
    Apply clean_text to a pandas Series or list of strings.
    """
    return [clean_text(t) for t in texts]
