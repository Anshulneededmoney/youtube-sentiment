# main.py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from googleapiclient.errors import HttpError

from src.config import YOUTUBE_API_KEY
from src.yt_fetch import fetch_comments
from src.clean import filter_relevant_comments, detect_lang, translate
from src.analysis import score_comments # This is your LLM scoring function

def get_counts(res: dict) -> list:
    """Helper function to extract sentiment counts from a result dictionary."""
    return [
        len(res.get("positive", [])),
        len(res.get("neutral", [])),
        len(res.get("negative", [])),
    ]

def plot_combined_results(results: dict, outdir: Path):
    """
    Plots a grouped bar chart comparing all four experimental modes.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    
    labels = ['Positive', 'Neutral', 'Negative']
    modes = list(results.keys()) # ['Full', 'Naive', 'Eng-Only', 'Hin-Only']
    
    counts = {mode: get_counts(results[mode]) for mode in modes}
    
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Create the bars for each mode
    rects1 = ax.bar(x - 1.5*width, counts[modes[0]], width, label=modes[0])
    rects2 = ax.bar(x - 0.5*width, counts[modes[1]], width, label=modes[1])
    rects3 = ax.bar(x + 0.5*width, counts[modes[2]], width, label=modes[2])
    rects4 = ax.bar(x + 1.5*width, counts[modes[3]], width, label=modes[3])

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Number of Comments')
    ax.set_title('Sentiment Analysis Comparison by Pipeline')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Pipeline Mode")

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)

    fig.tight_layout()
    
    plt.savefig(outdir / "combined_sentiment_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\n✅ Combined comparison plot saved to {outdir}/")
    plt.show()


def main():
    if not YOUTUBE_API_KEY:
        print(" Missing API key. Put `YOUTUBE_API_KEY=...` in your .env and run again.")
        return

    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Use your test URL
    max_comments = 200
    
    # --- [STEP 1/3] FETCH AND CLEAN (COMMON TO ALL) ---
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

    # --- [STEP 2/3] RUN ALL FOUR ANALYSIS PIPELINES ---
    print("\n[2/3] Running all four analysis pipelines…")
    
    all_results = {}
    
    # --- Mode 1: Full Pipeline (With Translation) ---
    print("\n--- Mode 1: Full (Translate Hindi) ---")
    full_comments = []
    hindi_count = 0
    for c in cleaned:
        lang = detect_lang(c)
        if lang == "hindi":
            hindi_count += 1
            full_comments.append(translate(c))
        else:
            full_comments.append(c)
    res_full = score_comments(full_comments)
    all_results["Full (Translate)"] = res_full
    print(f"    Analyzed {len(full_comments)} comments ({hindi_count} translated).")
    print(f"    Counts → +:{len(res_full['positive'])}  0:{len(res_full['neutral'])}  -:{len(res_full['negative'])}")

    # --- Mode 2: Naive Pipeline (No Translation) ---
    print("\n--- Mode 2: Naive (No Translate) ---")
    res_naive = score_comments(cleaned) # Analyze mixed-language list directly
    all_results["Naive (No Translate)"] = res_naive
    print(f"    Analyzed {len(cleaned)} comments (untranslated).")
    print(f"    Counts → +:{len(res_naive['positive'])}  0:{len(res_naive['neutral'])}  -:{len(res_naive['negative'])}")

    # --- Mode 3: English-Only (Filter out Hindi) ---
    print("\n--- Mode 3: English-Only (Filter Hindi) ---")
    eng_only_comments = []
    for c in cleaned:
        if detect_lang(c) == "english":
            eng_only_comments.append(c)
    res_eng_only = score_comments(eng_only_comments)
    all_results["English-Only"] = res_eng_only
    print(f"    Analyzed {len(eng_only_comments)} comments (Hindi comments ignored).")
    print(f"    Counts → +:{len(res_eng_only['positive'])}  0:{len(res_eng_only['neutral'])}  -:{len(res_eng_only['negative'])}")

    # --- Mode 4: Hindi-Only (Filter out English) ---
    print("\n--- Mode 4: Hindi-Only (Filter English) ---")
    hin_only_comments = []
    for c in cleaned:
        if detect_lang(c) == "hindi":
            hin_only_comments.append(translate(c)) # Translate the Hindi ones
    res_hin_only = score_comments(hin_only_comments)
    all_results["Hindi-Only (Translated)"] = res_hin_only
    print(f"    Analyzed {len(hin_only_comments)} comments (English comments ignored).")
    print(f"    Counts → +:{len(res_hin_only['positive'])}  0:{len(res_hin_only['neutral'])}  -:{len(res_hin_only['negative'])}")

    # --- [STEP 3/3] PLOT COMBINED RESULTS ---
    print("\n[3/3] Generating Combined Comparison Plot…")
    outdir = Path("outputs")
    plot_combined_results(all_results, outdir)


if __name__ == "__main__":
    main()
