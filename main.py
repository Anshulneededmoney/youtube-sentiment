# main.py
from pathlib import Path
import matplotlib.pyplot as plt
from googleapiclient.errors import HttpError

from src.config import YOUTUBE_API_KEY
from src.yt_fetch import fetch_comments
from src.clean import filter_relevant_comments
from src.analysis import score_comments


def plot_results(res, outdir: Path):
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
        print(" Missing API key. Put `YOUTUBE_API_KEY=...` in your .env and run again.")
        return

    # 👇 ask for inputs interactively
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    max_comments = 200
    preview = 8

    print("\n[1/4] Fetching comments…")
    try:
        raw = fetch_comments(YOUTUBE_API_KEY, video_url, max_total=max_comments)
    except HttpError as e:
        print(f" YouTube API error: {e}")
        return
    except Exception as e:
        print(f" Unexpected error: {e}")
        return

    print(f"   fetched: {len(raw)}")

    print("[2/4] Cleaning comments…")
    cleaned = filter_relevant_comments(raw)
    print(f"   kept: {len(cleaned)}")

    if not cleaned:
        print("No usable comments after cleaning.")
        return

    print("[3/4] Scoring sentiment…")
    res = score_comments(cleaned)
    avg = res["avg"]
    label = "Positive" if avg > 0.05 else "Negative" if avg < -0.05 else "Neutral"

    print("[4/4] Summary:")
    print(f"   Average polarity: {avg:.4f} → {label}")
    print(f"   Counts → +:{len(res['positive'])}  0:{len(res['neutral'])}  -:{len(res['negative'])}")

    print("\nPreview of cleaned comments:")
    for i, c in enumerate(cleaned[:preview], start=1):
        one = " ".join(c.splitlines()).strip()
        print(f"{i:>2}. {one[:200]}")

    print("\nSaving plots…")
    outdir = Path("outputs")
    plot_results(res, outdir)
    print(f"✅ Plots saved to {outdir}/")


if __name__ == "__main__":
    main()
