# src/clean.py
!pip install -U transformers
!pip install googletrans==3.1.0a0
from typing import List
import re
import emoji
import torch
from googletrans import Translator
translator = Translator()
from transformers import pipeline
pipe = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")
model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")

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

def detect_lang(text_to_classify_direct: str) -> str: # will return "hindi" or "english"

    # Tokenize the input
    inputs = tokenizer(text_to_classify_direct, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted class (logits)
    logits = outputs.logits

    hindi_prob = logits[0, 9]
    english_prob = logits[0, 13]
    
    if hindi_prob > english_prob:
        return "hindi"
    return "english"

def translate(text: str) -> str: # Romanized Hindi to English
    return translator.translate(text, src='hi', dest='en').text
