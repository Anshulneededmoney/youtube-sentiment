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
    
    # Create the bars
    rects = []
    bar_positions = [-1.5, -0.5, 0.5, 1.5]
    for i, mode in enumerate(modes):
        r = ax.bar(x + bar_positions[i]*width, ratios[mode], width, label=mode)
        rects.append(r)

    ax.set_ylabel('Ratio of Comments (0.0 to 1.0)')
    # --- 2. USE DYNAMIC TITLE ---
    ax.set_title(f'Normalized Sentiment Comparison: {title}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Pipeline Mode")
    ax.set_ylim(0, 1.0) 

    for r in rects:
        ax.bar_label(r, padding=3, fmt='%.2f')

    fig.tight_layout()
    
    # --- 3. USE DYNAMIC FILENAME ---
    plt.savefig(outdir / filename, dpi=150, bbox_inches="tight")
    print(f"\n✅ Comparison plot saved to {outdir / filename}")
    plt.show()


def run_all_scenarios(scoring_function, cleaned_comments):
    """
    A helper function to run the 4 analysis scenarios
    using the provided scoring_function.
    """
    all_results = {}

    # --- Mode 1: Full Pipeline (With Translation) ---
    print("\n--- Mode 1: Full (Translate Hindi) ---")
    full_comments = []
    hindi_count = 0
    for c in cleaned_comments:
        lang = detect_lang(c)
        if lang == "hindi":
            hindi_count += 1
            full_comments.append(translate(c))
        else:
            full_comments.append(c)
    res_full = scoring_function(full_comments)
    all_results["Full (Translate)"] = res_full
    print(f"    Analyzed {len(full_comments)} comments ({hindi_count} translated).")

    # --- Mode 2: Naive Pipeline (No Translation) ---
    print("\n--- Mode 2: Naive (No Translate) ---")
    res_naive = scoring_function(cleaned_comments)
    all_results["Naive (No Translate)"] = res_naive
    print(f"    Analyzed {len(cleaned_comments)} comments (untranslated).")

    # --- Mode 3: English-Only (Filter out Hindi) ---
    print("\n--- Mode 3: English-Only (Filter Hindi) ---")
    eng_only_comments = [c for c in cleaned_comments if detect_lang(c) == "english"]
    res_eng_only = scoring_function(eng_only_comments)
    all_results["English-Only"] = res_eng_only
    print(f"    Analyzed {len(eng_only_comments)} comments (Hindi comments ignored).")

    # --- Mode 4: Hindi-Only (Filter out English) ---
    print("\n--- Mode 4: Hindi-Only (Filter English) ---")
    hin_only_comments = [translate(c) for c in cleaned_comments if detect_lang(c) == "hindi"]
    res_hin_only = scoring_function(hin_only_comments)
    all_results["Hindi-Only (Translated)"] = res_hin_only
    print(f"    Analyzed {len(hin_only_comments)} comments (English comments ignored).")
    
    return all_results


def main():
    if not YOUTUBE_API_KEY:
        print(" Missing API key. Put `YOUTUBE_API_KEY=...` in your .env and run again.")
        return

    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Use your test URL
    max_comments = 200
    outdir = Path("outputs")
    
    # --- [STEP 1/3] FETCH AND CLEAN (Run once) ---
    print("\n[1/3] Fetching and Cleaning Comments…")
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

    # --- [STEP 2/3] RUN EXPERIMENTS FOR MODEL 1 (ROBERTA) ---
    if score_roberta:
        print("\n" + "="*50)
        print("   RUNNING EXPERIMENTS FOR RoBERTa (Multilingual)")
        print("="*50)
        roberta_results = run_all_scenarios(score_roberta, cleaned)
        plot_combined_results(roberta_results, outdir, 
                              "RoBERTa (Multilingual)", "comparison_roberta_ratio.png")
    
    # --- [STEP 3/3] RUN EXPERIMENTS FOR MODEL 2 (DISTILBERT) ---
    if score_distilbert:
        print("\n" + "="*50)
        print("   RUNNING EXPERIMENTS FOR DistilBERT (English-Only)")
        print("="*50)
        distilbert_results = run_all_scenarios(score_distilbert, cleaned)
        plot_combined_results(distilbert_results, outdir, 
                               "DistilBERT (English-Only)", "comparison_distilbert_ratio.png")


if __name__ == "__main__":
    main()
