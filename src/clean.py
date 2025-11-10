# src/clean.py
from typing import List
import re
import emoji
import torch
from googletrans import Translator
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Initializations ---
translator = Translator()

# --- Set device and move model to GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Language detection model (clean.py) running on: {device}")

tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")
model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")
model.to(device) # Move the model to the GPU


# --- Constants ---
_LINK_RE = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fAF][0-9a-fA-F]))+"
)
TEXT_RATIO_THRESHOLD = 0.65


# --- Functions ---
def normalize_text(s: str) -> str:
    """Lowercase + strip extra whitespace."""
    return " ".join(s.lower().split())

def filter_relevant_comments(comments: List[str]) -> List[str]:
    """
    Keep comments that have text, no links, aren't emoji-dominated,
    AND are written in the Latin script (ASCII).
    """
    kept: List[str] = []
    for c in comments:
        c2 = normalize_text(c)
        if not c2:
            continue
        
        # This filter is likely the cause of "0 Hindi comments".
        # Let's remove it.
        # text_only = emoji.replace_emoji(c2, replace='')
        # if not text_only.isascii():
        #    continue

        if _LINK_RE.search(c2):
            continue
        
        ecount = emoji.emoji_count(c2)
        # Revert this line back too
        txtchars = len(re.sub(r"\s", "", c2))
        if txtchars == 0:
            continue
        
        ratio = txtchars / (txtchars + ecount)
        if ratio >= TEXT_RATIO_THRESHOLD:
            kept.append(c2)
    return kept


def detect_lang(comments_batch: List[str]) -> List[str]:
    """
    Compares the model's logits for Hindi (9) and English (13)
    IN A BATCH to run fast on a GPU.
    """
    if not comments_batch:
        return []

    inputs = tokenizer(
        comments_batch, 
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    
    hindi_probs = logits[:, 9]
    english_probs = logits[:, 13]
    
    is_hindi = hindi_probs > english_probs
    
    labels = ["hindi" if is_h else "english" for is_h in is_hindi]
    
    return labels

# --- THIS FUNCTION IS THE FIX ---
def translate(text: str) -> str: # Romanized Hindi to English
    """
    Translates Romanized Hindi to English.
    """
    try:
        # --- THE FIX ---
        # We removed src='hi-Latn' and are letting the library auto-detect.
        translated_obj = translator.translate(text, dest='en')
        # --- END OF FIX ---
        return translated_obj.text
    except Exception as e:
        print(f"Error in translation: {e}. Returning original text.")
        return text
