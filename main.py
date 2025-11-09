# main.py
from pathlib import Path
import matplotlib.pyplot as plt
from googleapiclient.errors import HttpError

from src.config import YOUTUBE_API_KEY
from src.yt_fetch import fetch_comments
# --- MODIFIED: Import the new functions ---
from src.clean import filter_relevant_comments, detect_lang, translate
from src.analysis import score_comments


def plot_results(res, outdir: Path):
    # ... (This function remains unchanged) ...
    outdir.mkdir(parents=True, exist_ok=True)

    # Histogram of polarity
    plt.figure()
    plt.hist(res["polarity"], bins=21, range=(-1, 1))
    plt.title("Comment Sentiment Polarity (VADER compound)")
    plt.xlabel("Polarity (−1 … 1)")
    plt.ylabel("Frequency")
    plt.savefig(outdir / "polarity_hist.png", dpi=150, bbox_inches="tight")

    # Bar chart of class counts
    labels = ["Positive", "Neutral", "Negative"]
    counts = [len(res["positive"]), len(res["neutral"]), len(res["negative"])]

    plt.figure()
    plt.bar(labels, counts)
    plt.title("Sentiment Class Counts")
    plt.ylabel("Count")
    plt.savefig(outdir / "class_counts.png", dpi=150, bbox_inches="tight")

    plt.show()


def main():
    if not YOUTUBE_API_KEY:
        print(" Missing API key. Put ⁠ YOUTUBE_API_KEY=... ⁠ in your .env and run again.")
        return

    # 👇 ask for inputs interactively
    video_url = "https://www.youtube.com/watch?v=WzvURhaDZqI&pp=ygUHY2FtcHVzeNIHCQkDCgGHKiGM7w%3D%3D"
    max_comments = 200
    preview = 8

    print("\n[1/5] Fetching comments…") # <-- Renumbered to 5 steps
    try:
        raw = fetch_comments(YOUTUBE_API_KEY, video_url, max_total=max_comments)
    except HttpError as e:
        print(f" YouTube API error: {e}")
        return
    except Exception as e:
        print(f" Unexpected error: {e}")
        return

    print(f"    fetched: {len(raw)}")

    print("[2/5] Cleaning comments…") # <-- Renumbered
    cleaned = filter_relevant_comments(raw)
    print(f"    kept: {len(cleaned)}")

    if not cleaned:
        print("No usable comments after cleaning.")
        return

    # --- NEW STEP: Language Detection & Translation ---
    print("[3/5] Translating Romanized Hindi comments…")
    final_comments_for_analysis = []
    hindi_count = 0
    for comment in cleaned:
        lang = detect_lang(comment)
        if lang == "hindi":
            hindi_count += 1
            # Translate Hindi comment to English
            translated_comment = translate(comment)
            if translated_comment: # Only add if translation was successful
                final_comments_for_analysis.append(translated_comment)
        else:
            # Keep the English comment as is
            final_comments_for_analysis.append(comment)
    
    print(f"    found and translated {hindi_count} Hindi comments.")
    print(f"    total comments for analysis: {len(final_comments_for_analysis)}")
    # --- END OF NEW STEP ---

    if not final_comments_for_analysis:
        print("No usable comments left after translation step.")
        return

    print("[4/5] Scoring sentiment…") # <-- Renumbered
    # --- MODIFIED: Use the newly translated list ---
    res = score_comments(final_comments_for_analysis)
    avg = res["avg"]
    label = "Positive" if avg > 0.05 else "Negative" if avg < -0.05 else "Neutral"

    print("[5/5] Summary:") # <-- Renumbered
    print(f"    Average polarity: {avg:.4f} → {label}")
    print(f"    Counts → +:{len(res['positive'])}  0:{len(res['neutral'])}  -:{len(res['negative'])}")

    print("\nPreview of final comments (after translation):")
    # --- MODIFIED: Show the final translated comments ---
    for i, c in enumerate(final_comments_for_analysis[:preview], start=1):
        one = " ".join(c.splitlines()).strip()
        print(f"{i:>2}. {one[:200]}")

    print("\nSaving plots…")
    outdir = Path("outputs")
    plot_results(res, outdir)
    print(f"✅ Plots saved to {outdir}/")


if _name_ == "_main_":
    main()
