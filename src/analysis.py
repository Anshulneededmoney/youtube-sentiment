# src/analysis.py
from typing import Dict, List
from transformers import pipeline
import torch

# --- NEW: Initialize the LLM pipeline ---
# Check if a GPU is available, otherwise use CPU
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use {'cuda:0' if device == 0 else 'cpu'}")

# Load the fine-tuned RoBERTa model
# This will download the model the first time it's run
pipe = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=device
)

def score_comments(comments: List[str]) -> Dict[str, object]:
    """
    Returns dict with polarity list, positive/negative/neutral splits, and average,
    using a Transformer LLM (RoBERTa).
    """
    polarity: List[float] = []
    pos: List[str] = []
    neg: List[str] = []
    neu: List[str] = []

    # Process comments in batches for massive speedup
    # The pipeline is much faster when it processes many comments at once
    try:
        results = pipe(comments, truncation=True, max_length=512)
    except Exception as e:
        print(f"Error during pipeline batch processing: {e}")
        # Fallback to one-by-one if batch fails
        results = []
        for c in comments:
            try:
                results.append(pipe(c, truncation=True, max_length=512)[0])
            except Exception as e_inner:
                print(f"Error processing comment: {c[:50]}... | Error: {e_inner}")
                results.append(None) # Add a placeholder

    for i, c in enumerate(comments):
        result = results[i]
        if not result:
            continue # Skip if this comment failed

        label = result['label'].lower()
        score = result['score']

        # Convert the LLM's output to the format main.py expects
        # We create our own 'compound' score for the histogram
        if label == 'positive':
            polarity_score = score # e.g., 0.9
            pos.append(c)
        elif label == 'negative':
            polarity_score = -1 * score # e.g., -0.9
            neg.append(c)
        else: # 'neutral'
            polarity_score = 0.0 # Neutral is 0.0
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
