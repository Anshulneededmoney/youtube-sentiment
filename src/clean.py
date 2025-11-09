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
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
TEXT_RATIO_THRESHOLD = 0.65


# --- Functions ---
def normalize_text(s: str) -> str:
    """Lowercase + strip extra whitespace."""
    return " ".join(s.lower().split())

def filter_relevant_comments(comments: List[str]) -> List[str]:
    """
    Keep comments that have text, no links, and aren't emoji-dominated.
    (This version has no .isascii() filter)
    """
    kept: List[str] = []
    for c in comments:
        c2 = normalize_text(c)
        if not c2:
            continue

        # NO .isascii() CHECK HERE

        if _LINK_RE.search(c2):
            continue
        
        ecount = emoji.emoji_count(c2)
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

    # Tokenize the entire batch at once
    inputs = tokenizer(
        comments_batch, 
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True  # Pad to the longest comment in the batch
    )

    # Move the input tensors to the same device as the model (the GPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    
    # Get probabilities for only Hindi (9) and English (13) for the whole batch
    hindi_probs = logits[:, 9]
    english_probs = logits[:, 13]
    
    # Compare probabilities
    is_hindi = hindi_probs > english_probs # This is a boolean Tensor
    
    # Convert boolean tensor back to a list of strings
    labels = ["hindi" if is_h else "english" for is_h in is_hindi]
    
    return labels

def translate(text: str) -> str: # Romanized Hindi to English
    """
    Translates Romanized Hindi ("hi-Latn") to English ("en").
    """
    try:
        translated_obj = translator.translate(text, src='hi-Latn', dest='en')
        return translated_obj.text
    except Exception as e:
        print(f"Error in translation: {e}. Returning original text.")
        return text
