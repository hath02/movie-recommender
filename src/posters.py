import os
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API_KEY = os.environ.get("TMDB_API_KEY")
CACHE_FILE = "models/poster_cache.json"
IMG_BASE = "https://image.tmdb.org/t/p/w500"

try:
    with open(CACHE_FILE, "r") as f:
        _cache = json.load(f)
except FileNotFoundError:
    _cache = {}

def fetch_poster(tmdb_id):
    if not tmdb_id or not API_KEY:
        return ""
    key = str(int(float(tmdb_id)))
    if key in _cache:
        return _cache[key]

    url = f"https://api.themoviedb.org/3/movie/{key}?api_key={API_KEY}"
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                path = json.load(r).get("poster_path")
            poster = f"{IMG_BASE}{path}" if path else ""
            _cache[key] = poster
            return poster
        except Exception as e:
            print(f"poster error (tmdb {key}, try {attempt + 1}):", e)
    return ""
    

def add_posters(movies):
    """movies: list of dicts with 'tmdb_id'. Add a 'poster' key to each dict."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        urls = list(pool.map(fetch_poster, [m["tmdb_id"] for m in movies]))
    for m, url in zip(movies, urls):
        m["poster"] = url
    try:
        os.makedirs("models", exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(_cache, f)
    except Exception:
        pass
    return movies