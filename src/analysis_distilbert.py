from typing import Dict, List
from transformers import pipeline
import torch

# --- NEW: Initialize the ENGLISH-ONLY LLM pipeline ---
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use {'cuda:0' if device == 0 else 'cpu'}")

# Load the English-only model
# This model was trained ONLY on English movie reviews (SST-2)
print("Loading English-only sentiment model (distilbert-base-uncased-finetuned-sst-2-english)...")
pipe = pipeline(
    "sentiment-analysis", 
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)
print("English-only model loaded.")


def score_comments(comments: List[str]) -> Dict[str, object]:
    """
    Returns dict with polarity list, positive/negative/neutral splits,
    using a Transformer LLM (DistilBERT) trained ONLY on English.
    """
    polarity: List[float] = []
    pos: List[str] = []
    neg: List[str] = []
    neu: List[str] = []

    # Process comments in batches
    try:
        results = pipe(comments, truncation=True, max_length=512)
    except Exception as e:
        print(f"Error during pipeline batch processing: {e}")
        return { "polarity": [], "positive": [], "negative": [], "neutral": [], "avg": 0.0 }


    for i, c in enumerate(comments):
        result = results[i]
        label = result['label'].upper()
        score = result['score']

        # Convert the LLM's output to the format main.py expects
        # This model only has "POSITIVE" and "NEGATIVE" labels.
        # It will misclassify Romanized Hindi as one of these.
        if label == 'POSITIVE':
            polarity_score = score
            pos.append(c)
        elif label == 'NEGATIVE':
            polarity_score = -1 * score
            neg.append(c)
        else:
            # This model does not output "NEUTRAL", so this will be empty
            polarity_score = 0.0
            neu.append(c)
            
        polarity.append(polarity_score)

    avg = (sum(p for p in polarity) / len(polarity)) if polarity else 0.0
    return {
        "polarity": polarity,
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "avg": avg,
    }
