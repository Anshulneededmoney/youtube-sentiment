import os
from dotenv import load_dotenv

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not YOUTUBE_API_KEY:
    print("⚠️  No YOUTUBE_API_KEY found in .env (we can still run a dry test).")
