# 🎥 YouTube Sentiment Analysis

Analyze the sentiment of YouTube video comments using Python, VADER sentiment analysis, and the YouTube Data API v3.

This project automatically fetches comments from any YouTube video, cleans them by removing links and emojis, and evaluates how positive, negative, or neutral the overall comment section is.  
It then generates sentiment plots and saves them to an `outputs/` folder.

---

## 🚀 Features
- Fetch comments from any YouTube video using the YouTube Data API v3  
- Clean comments (remove links, emojis, and empty text)  
- Compute sentiment scores using **VADER**  
- Visualize polarity distribution and sentiment class counts  
- Save results & plots automatically  

---

## 🧩 Project Structure
youtube-sentiment/
├── src/
│ ├── config.py # Load API key from .env
│ ├── yt_fetch.py # Fetch YouTube comments
│ ├── clean.py # Clean comments
│ ├── analysis.py # Sentiment analysis
│ └── init.py
├── main.py # Main entry point
├── requirements.txt # Python dependencies
├── .env # Your API key (not committed)
├── .gitignore
└── outputs/ # Saved plots

## ⚙️ Setup Instructions

### 1️⃣ Clone this repository
```bash
git clone git@github.com:anshulneededmoney/youtube-sentiment.git
cd youtube-sentiment

python -m venv .venv
.venv\Scripts\activate      # (Windows)
# or
source .venv/bin/activate   # (Mac/Linux)

pip install -r requirements.txt

Create a .env file in the project root:
YOUTUBE_API_KEY=your_api_key_here

▶️ Usage

You can modify the hardcoded values in main.py:
video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
max_comments = 200
preview = 8

then run- python main.py

