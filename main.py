#main.py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from googleapiclient.errors import HttpError

from src.config import YOUTUBE_API_KEY
from src.yt_fetch import fetch_comments
from src.clean import filter_relevant_comments, detect_lang, translate

# --- 1. IMPORT BOTH SCORING FUNCTIONS ---
try:
    from src.analysis_roberta import score_comments as score_roberta
    print("Successfully imported RoBERTa (multilingual) model.")
except ImportError:
    print("Could not import RoBERTa model.")
    score_roberta = None

try:
    from src.analysis_distilbert import score_comments as score_distilbert
    print("Successfully imported DistilBERT (English-only) model.")
except ImportError:
    print("Could not import DistilBERT model.")
    score_distilbert = None


def get_counts(res: dict) -> list:
    """Helper function to extract sentiment counts from a result dictionary."""
    return [
        len(res.get("positive", [])),
        len(res.get("neutral", [])),
        len(res.get("negative", [])),
    ]

def plot_combined_results(results: dict, outdir: Path, title: str, filename: str):
    """
    Plots a grouped bar chart comparing all four experimental modes
    using RATIOS instead of raw counts.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    
    labels = ['Positive', 'Neutral', 'Negative']
    modes = list(results.keys())
    
    # Calculate ratios
    ratios = {}
    for mode in modes:
        counts = get_counts(results[mode])
        total = sum(counts)
        if total == 0:
            ratios[mode] = [0.0, 0.0, 0.0]
        else:
            ratios[mode] = [c / total for c in counts]
    
    x = np.arange(len(labels))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    rects = []
    bar_positions = [-1.5, -0.5, 0.5, 1.5]
    for i, mode in enumerate(modes):
        r = ax.bar(x + bar_positions[i]*width, ratios[mode], width, label=mode)
        rects.append(r)

    ax.set_ylabel('Ratio of Comments (0.0 to 1.0)')
    ax.set_title(f'Normalized Sentiment Comparison: {title}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Pipeline Mode")
    ax.set_ylim(0, 1.0) 

    for r in rects:
        ax.bar_label(r, padding=3, fmt='%.2f')

    fig.tight_layout()
    
    plt.savefig(outdir / filename, dpi=150, bbox_inches="tight")
    print(f"\n✅ Comparison plot saved to {outdir / filename}")
    plt.show()


def main():
    if not YOUTUBE_API_KEY:
        print(" Missing API key. Put `YOUTUBE_API_KEY=...` in your .env and run again.")
        return

    video_url = "https.www.youtube.com/watch?v=dQw4w9WgXcQ" # Use your test URL
    max_comments = 200
    outdir = Path("outputs")
    
    # --- [STEP 1/4] FETCH AND CLEAN (Run once) ---
    print("\n[1/4] Fetching and Cleaning Comments…")
    try:
        raw = fetch_comments(YOUTUBE_API_KEY, video_url, max_total=max_comments)
    except HttpError as e:
        print(f" YouTube API error: {e}")
        return
    except Exception as e:
        print(f" Unexpected error: {e}")
        return
    
    cleaned = filter_relevant_comments(raw)
    if not cleaned:
        print("No usable comments found after cleaning.")
        return
    print(f"    Fetched: {len(raw)}, Kept after cleaning: {len(cleaned)}")

    # --- [STEP 2/4] PRE-PROCESS & BUILD SCENARIO LISTS (THE OPTIMIZATION) ---
    # We build all 4 lists in a single, efficient loop.
    
    print("\n[2/4] Detecting language and translating (ONE TIME PASS)...")
    
    comments_full_translate = []
    comments_naive = cleaned # This is just the original list
    comments_eng_only = []
    comments_hin_only_translated = []
    
    hindi_count = 0
    
    # This single loop does all the slow work (LLM detection, API calls)
    for c in cleaned:
        lang = detect_lang(c)
        if lang == "hindi":
            hindi_count += 1
            translated_c = translate(c)
            
            # Add to lists that use the translated version
            comments_full_translate.append(translated_c)
            comments_hin_only_translated.append(translated_c)
        else: # lang is "english"
            # Add to lists that use the original English version
            comments_full_translate.append(c)
            comments_eng_only.append(c)

    print(f"    Analyzed {len(cleaned)} comments, found {hindi_count} Hindi comments.")

    # --- [STEP 3/4] RUN EXPERIMENTS (This part is fast) ---
    print("\n[3/4] Running sentiment analysis on all 8 scenarios...")
    
    all_results_roberta = {}
    all_results_distilbert = {}

    # --- Run RoBERTa on the 4 pre-processed lists ---
    if score_roberta:
        print("--- Scoring with RoBERTa (Multilingual) ---")
        all_results_roberta["Full (Translate)"] = score_roberta(comments_full_translate)
        all_results_roberta["Naive (No Translate)"] = score_roberta(comments_naive)
        all_results_roberta["English-Only"] = score_roberta(comments_eng_only)
        all_results_roberta["Hindi-Only (Translated)"] = score_roberta(comments_hin_only_translated)

    # --- Run DistilBERT on the 4 pre-processed lists ---
    if score_distilbert:
        print("--- Scoring with DistilBERT (English-Only) ---")
        all_results_distilbert["Full (Translate)"] = score_distilbert(comments_full_translate)
        all_results_distilbert["Naive (No Translate)"] = score_distilbert(comments_naive)
        all_results_distilbert["English-Only"] = score_distilbert(comments_eng_only)
        all_results_distilbert["Hindi-Only (Translated)"] = score_distilbert(comments_hin_only_translated)

    # --- [STEP 4/4] PLOT RESULTS ---
    print("\n[4/4] Generating comparison plots...")

    if score_roberta:
        plot_combined_results(all_results_roberta, outdir, 
                              "RoBERTa (Multilingual)", "comparison_roberta_ratio.png")
    
    if score_distilbert:
        plot_combined_results(all_results_distilbert, outdir, 
                               "DistilBERT (English-Only)", "comparison_distilbert_ratio.png")


if __name__ == "__main__":
    main()
