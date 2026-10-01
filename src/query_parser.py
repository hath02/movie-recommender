import re
import math
import torch
import torch.nn.functional as F
import difflib
from src.recommendation import fetch_all, get_all_movies, get_movie_rating_stats

GENRE_KEYWORDS = {
    "action": "Action", "funny": "Comedy", "comedy": "Comedy",
    "scary": "Horror", "horror": "Horror", "romantic": "Romance",
    "romance": "Romance", "sci-fi": "Sci-Fi", "scifi": "Sci-Fi",
    "drama": "Drama", "animated": "Animation", "animation": "Animation",
    "thriller": "Thriller", "crime": "Crime", "adventure": "Adventure",
    "fantasy": "Fantasy", "war": "War", "mystery": "Mystery",
    "documentary": "Documentary", "family": "Children",
}

FILLER = {
    "i", "want", "like", "but", "with", "some", "more", "less", "similar",
    "to", "movie", "movies", "film", "something", "me", "show", "give",
    "a", "an", "and", "also", "watch",
}

_MF = {"emb": None, "m2i": None}

def load_mf(path="models/movy_mf.pt"):
    if _MF["emb"] is None:
        try:
            ck = torch.load(path, map_location="cpu")
            w = ck["model_state_dict"]["movie_embedding.weight"]
            _MF["emb"] = F.normalize(w, dim=1)   # unit length, so dot product = cosine
            _MF["m2i"] = ck["movie_to_index"]
            
        except Exception as e:
            print(f"(No trained model loaded: {e})")
            _MF["emb"] = False
            
    return _MF

def embedding_sim(a, b):
    emb, m2i = _MF["emb"], _MF["m2i"]
    if emb is False or emb is None or a not in m2i or b not in m2i:
        return None
    return max(0.0, float(emb[m2i[a]] @ emb[m2i[b]]))

def find_movie(text_lower, movies, total):
    # 1) exact match: longest title wins, then the most-rated one
    best_key, best_id = (0, 0), None
    for movie_id, title, genres, _ in movies:
        name = clean_title(title)
        if len(name) < 3:
            continue
        if re.search(rf"\b{re.escape(name)}\b", text_lower):
            key = (len(name), total.get(movie_id, 0))
            if key > best_key:
                best_key, best_id = key, movie_id
    if best_id is not None:
        return best_id

    # 2) fuzzy match (handles typos), only against reasonably popular movies
    titles = {}
    for movie_id, title, _, _ in movies:
        count = total.get(movie_id, 0)
        if count < 10:
            continue
        name = clean_title(title)
        if len(name) < 4:
            continue
        if name not in titles or count > total.get(titles[name], 0):
            titles[name] = movie_id

    tokens = re.findall(r"[a-z0-9']+", text_lower)
    skip = FILLER | set(GENRE_KEYWORDS)
    best_ratio, best_id = 0.0, None

    for size in range(1, 5):
        for i in range(len(tokens) - size + 1):
            gram = tokens[i:i + size]
            if gram[0] in skip or gram[-1] in skip:
                continue
            phrase = " ".join(gram)
            if len(phrase) < 4:
                continue
            for name in difflib.get_close_matches(phrase, titles, n=1, cutoff=0.8):
                ratio = difflib.SequenceMatcher(None, phrase, name).ratio()
                if ratio > best_ratio:
                    best_ratio, best_id = ratio, titles[name]

    return best_id


def parse_query(text, movies, total):
    text_lower = text.lower()
    movie_id = find_movie(text_lower, movies, total)

    words = set(re.findall(r"[a-z\-]+", text_lower))
    wanted = {g for word, g in GENRE_KEYWORDS.items() if word in words}
    return movie_id, wanted
                
def clean_title(title):
    # "Interstellar (2014)" -> "interstellar"
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", title)
    return title.lower().strip()

def get_fans(movie_id):
    # try a strict threshold first, then relax it for less popular movies
    rows = []
    for min_fans in (30, 10, 3, 1):
        rows = fetch_all("""
            SELECT r2.movie_id, COUNT(*) AS fans
            FROM ratings r1
            JOIN ratings r2 ON r1.user_id = r2.user_id
            WHERE r1.movie_id = ? AND r1.rating >= 4.0
              AND r2.rating >= 4.0 AND r2.movie_id != ?
            GROUP BY r2.movie_id
            HAVING fans >= ?
            ORDER BY fans DESC
            LIMIT 500
        """, (movie_id, movie_id, min_fans))
        if len(rows) >= 30:
            break
    return rows


def genre_fallback(movie_id, wanted, info, total, avg, n):
    # used when almost nobody rated the movie: rank by genre overlap + quality
    base = set(info[movie_id][2].split("|")) | wanted
    scored = []
    for mid, (_, title, genres, _) in info.items():
        if mid == movie_id or total.get(mid, 0) < 50:
            continue
        g = set(genres.split("|"))
        if wanted and not (wanted & g):
            continue
        jaccard = len(base & g) / len(base | g)
        c = total[mid]
        a = float(avg.get(mid, 3.5))
        wr = (c / (c + 100)) * a + (100 / (c + 100)) * 3.5
        scored.append((0.6 * jaccard + 0.4 * (wr / 5), title))
    scored.sort(reverse=True)
    return [t for _, t in scored[:n]]

def recommend_from_text(text, n=16):
    movies = get_all_movies()
    load_mf()
    stats = get_movie_rating_stats()
    total = dict(zip(stats["movieId"], stats["rating_count"]))
    avg = dict(zip(stats["movieId"], stats["average_rating"]))

    movie_id, wanted = parse_query(text, movies, total)
    if movie_id is None:
        return []

    info = {m[0]: m for m in movies}
    print(f"Matched movie: {info[movie_id][1]}  |  genres wanted: {wanted or 'none'}")

    rows = get_fans(movie_id)

    candidates = []
    for mid, fans in rows:
        if mid not in info:
            continue

        genres = set(info[mid][2].split("|"))
        match = 1.0 if (wanted & genres) else 0.0
        cf = fans / math.sqrt(total.get(mid, 1))
        c = total.get(mid, 0)
        a = float(avg.get(mid, 3.5))
        wr = (c / (c + 100)) * a + (100 / (c + 100)) * 3.5
        candidates.append((mid, cf, wr, match))

    if not candidates:
        return genre_fallback(movie_id, wanted, info, total, avg, n)

    if wanted:
        matched = [c for c in candidates if c[3]]
        if len(matched) >= n:
            candidates = matched

    max_cf = max(c[1] for c in candidates)
    scored = []
    for mid, cf, wr, match in candidates:
        sim = embedding_sim(movie_id, mid)
        if sim is None:
            score = 0.6 * (cf / max_cf) + 0.25 * match + 0.15 * (wr / 5)
        else:
            score = 0.45 * (cf / max_cf) + 0.25 * sim + 0.2 * match + 0.1 * (wr / 5)
        scored.append((score, info[mid][1])) 

    scored.sort(reverse=True)
    
    results = [title for _, title in scored[:n]]

    if len(results) < n:
        extra = genre_fallback(movie_id, wanted, info, total, avg, n * 2)
        
        for t in extra:
            if t not in results:
                results.append(t)
            
            if len(results) >= n:
                break
            
    return results


if __name__ == "__main__":
    while True:
        text = input("\nWhat do you want to watch? (q to quit): ")
        if text.lower() == "q":
            break

        results = recommend_from_text(text)
        if not results:
            print("No results. Try a popular movie title.")

        for i, title in enumerate(results, 1):
            print(f"{i}. {title}")