# src/analysis.py
from typing import Dict, List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

TH_POS = 0.05
TH_NEG = -0.05

def score_comments(comments: List[str]) -> Dict[str, object]:
    """
    Returns dict with polarity list, positive/negative/neutral splits, and average.
    """
    sia = SentimentIntensityAnalyzer()
    polarity: List[float] = []
    pos: List[str] = []
    neg: List[str] = []
    neu: List[str] = []

    for c in comments:
        comp = sia.polarity_scores(c)["compound"]
        polarity.append(comp)
        if comp > TH_POS:
            pos.append(c)
        elif comp < TH_NEG:
            neg.append(c)
        else:
            neu.append(c)

    avg = (sum(p for p in polarity) / len(polarity)) if polarity else 0.0
    return {
        "polarity": polarity,
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "avg": avg,
    }
