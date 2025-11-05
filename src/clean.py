# src/clean.py
from typing import List
import re
import emoji

_LINK_RE = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
TEXT_RATIO_THRESHOLD = 0.65  # keep when text dominates emojis

def normalize_text(s: str) -> str:
    """Lowercase + strip extra whitespace."""
    return " ".join(s.lower().split())

def filter_relevant_comments(comments: List[str]) -> List[str]:
    """
    Keep comments that have text, no links, and aren't emoji-dominated.
    """
    kept: List[str] = []
    for c in comments:
        c2 = normalize_text(c)
        if not c2:
            continue
        if _LINK_RE.search(c2):
            continue
        # estimate if it's mostly emoji
        ecount = emoji.emoji_count(c2)
        txtchars = len(re.sub(r"\s", "", c2))
        if txtchars == 0:
            continue
        ratio = txtchars / (txtchars + ecount)
        if ratio >= TEXT_RATIO_THRESHOLD:
            kept.append(c2)
    return kept
