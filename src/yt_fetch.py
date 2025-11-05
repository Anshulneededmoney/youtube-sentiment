from googleapiclient.discovery import build
import urllib.parse as up
import re

def get_video_id_from_url(url: str) -> str:
    """Extracts the YouTube video ID from common URL formats."""
    parsed = up.urlparse(url)
    qs = up.parse_qs(parsed.query)
    if "v" in qs and qs["v"]:
        return qs["v"][0]  # e.g. https://www.youtube.com/watch?v=VIDEOID
    # fallback: try youtu.be/<id> or last 11 chars
    path_id = parsed.path.strip("/").split("/")[-1]
    m = re.match(r"^[A-Za-z0-9_-]{11}$", path_id)
    return path_id if m else url[-11:]

def fetch_comments(api_key: str, video_url: str, max_total: int = 100):
    """
    Returns up to `max_total` top-level comments (plain text).
    Requires a valid YouTube Data API v3 key.
    """
    youtube = build("youtube", "v3", developerKey=api_key)
    video_id = get_video_id_from_url(video_url)

    comments = []
    next_token = None

    while len(comments) < max_total:
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_token
        )
        res = req.execute()
        for item in res.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(snippet["textDisplay"])
            if len(comments) >= max_total:
                break
        next_token = res.get("nextPageToken")
        if not next_token:
            break

    return comments
