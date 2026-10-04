import os
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from database.database import fetch_all, connect

API_KEY = os.environ["TMDB_API_KEY"]
TOP_N = 500            # test with 500 first, then raise to 8000
CACHE = "models/credits_cache.json"

try:
    with open(CACHE) as f:
        cache = json.load(f)
except Exception:
    cache = {}


def fetch(item):
    movie_id, tmdb_id = item
    key = str(movie_id)
    if key in cache:
        return key, cache[key]
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}/credits?api_key={API_KEY}"
    err = None
    for _ in range(2):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                d = json.load(r)
            people = [[p["id"], p["name"], "director"]
                      for p in d.get("crew", []) if p.get("job") == "Director"]
            people += [[p["id"], p["name"], "actor"] for p in d.get("cast", [])[:8]]
            return key, people
        except Exception as e:
            err = e
    print("failed:", movie_id, err)
    return key, None


movies = fetch_all(f"""
    SELECT s.movie_id, l.tmdb_id
    FROM movie_stats s
    JOIN links l ON l.movie_id = s.movie_id
    WHERE l.tmdb_id IS NOT NULL
    ORDER BY s.rating_count DESC
    LIMIT {TOP_N}
""")
print(f"{len(movies):,} movies to process")

CHUNK = 400
for i in range(0, len(movies), CHUNK):
    with ThreadPoolExecutor(max_workers=8) as pool:
        for key, people in pool.map(fetch, movies[i:i + CHUNK]):
            if people is not None:
                cache[key] = people
    os.makedirs("models", exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(cache, f)
    print(f"{min(i + CHUNK, len(movies)):,} / {len(movies):,}")

# write to the database (INSERT IGNORE makes reruns safe)
people_rows, mp_rows = {}, []
for key, people in cache.items():
    for pid, name, role in people:
        people_rows[pid] = name.strip()
        mp_rows.append((int(key), pid, role))

conn = connect()
cur = conn.cursor()
cur.executemany("INSERT IGNORE INTO people (person_id, name) VALUES (?, ?)",
                list(people_rows.items()))
for i in range(0, len(mp_rows), 20000):
    cur.executemany(
        "INSERT IGNORE INTO movie_people (movie_id, person_id, role) VALUES (?, ?, ?)",
        mp_rows[i:i + 20000])
conn.commit()
cur.close()
conn.close()
print(f"{len(people_rows):,} people, {len(mp_rows):,} links saved")