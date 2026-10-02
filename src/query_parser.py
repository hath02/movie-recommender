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

# word -> (genres to boost, genres to avoid)
MOODS = {
    "dark":        ({"Crime", "Thriller", "Horror"}, {"Comedy", "Children", "Animation"}),
    "darker":      ({"Crime", "Thriller", "Horror"}, {"Comedy", "Children", "Animation"}),
    "gritty":      ({"Crime", "Thriller", "War"}, {"Comedy", "Children"}),
    "serious":     ({"Drama", "War", "Crime"}, {"Comedy", "Children"}),
    "light":       ({"Comedy", "Romance", "Children"}, {"Horror", "War"}),
    "lighter":     ({"Comedy", "Romance", "Children"}, {"Horror", "War"}),
    "lighthearted":({"Comedy", "Romance", "Children"}, {"Horror", "War"}),
    "feel-good":   ({"Comedy", "Romance", "Children"}, {"Horror", "War"}),
    "happy":       ({"Comedy", "Romance"}, {"War", "Horror"}),
    "uplifting":   ({"Comedy", "Romance", "Children"}, {"War", "Horror"}),
    "sad":         ({"Drama", "War"}, set()),
    "depressing":  ({"Drama", "War"}, set()),
    "emotional":   ({"Drama", "Romance"}, set()),
    "cartoon":     ({"Animation", "Children"}, set()),
    "cartoons":    ({"Animation", "Children"}, set()),
    "kids":        ({"Animation", "Children", "Family"}, {"Horror", "Crime"}),
    "scarier":     ({"Horror", "Thriller"}, set()),
    "creepy":      ({"Horror", "Thriller", "Mystery"}, set()),
    "violent":     ({"Action", "Crime", "War", "Horror"}, set()),
    "intense":     ({"Thriller", "Action", "War"}, set()),
    "exciting":    ({"Action", "Adventure", "Thriller"}, set()),
    "epic":        ({"Adventure", "War", "Fantasy"}, set()),
    "smart":       ({"Mystery", "Sci-Fi", "Thriller"}, set()),
    "twisty":      ({"Mystery", "Thriller"}, set()),
    "mindbending": ({"Mystery", "Sci-Fi", "Thriller"}, set()),
    "mind-bending":({"Mystery", "Sci-Fi", "Thriller"}, set()),
    "space":       ({"Sci-Fi"}, set()),
    "futuristic":  ({"Sci-Fi"}, set()),
    "magical":     ({"Fantasy", "Adventure"}, set()),
    "funnier":     ({"Comedy"}, set()),
    "romantic":    ({"Romance"}, set()),
}

# every keyword -> (boost set, avoid set)
KEYWORDS = {w: ({g}, set()) for w, g in GENRE_KEYWORDS.items()}
KEYWORDS.update(MOODS)

FILLER = {
    "i", "want", "like", "but", "with", "some", "more", "less", "similar",
    "to", "movie", "movies", "film", "something", "me", "show", "give",
    "a", "an", "and", "also", "watch",
}

NEGATORS = {"no", "not", "without", "less", "avoid", "hate", "except",
            "nothing", "minus", "fewer", "don", "dont", "never"}

STOP = {"the", "of", "in", "for", "on", "it", "is", "that", "this", "are",
        "can", "at", "as", "be", "just", "very", "too", "bit", "little",
        "any", "then", "than", "from", "have", "has", "was", "were"}

_MF = {"emb": None, "m2i": None}

_TE = {"emb": None, "m2i": None}

_TAG_NAMES = set()
_TAGS = {}  # tags -> {movie_id: score}, filled lazily


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


def load_tag_emb(path="models/tag_emb.pt"):
    if _TE["emb"] is None:
        try:
            ck = torch.load(path, map_location="cpu")
            _TE["emb"] = ck["emb"]   # unit length, so dot product = cosine
            _TE["row"] = {m: i for i, m in enumerate(ck["movie_ids"])}
            
        except Exception as e:
            print(f"(No tag embedding loaded: {e})")
            _TE["emb"] = False
            
    return _TE


def get_vec(movie_id):
    # prefer the real MF embedding, otherwise the one predicted from tags
    if _MF["emb"] is not None and _MF["emb"] is not False and movie_id in _MF["m2i"]:
        return _MF["emb"][_MF["m2i"][movie_id]]
    if _TE["emb"] is not None and _TE["emb"] is not False and movie_id in _TE["row"]:
        return _TE["emb"][_TE["row"][movie_id]]
    return None


def load_tags():
    if not _TAG_NAMES:
        rows = fetch_all("""
            SELECT tag 
            FROM movie_tag_scores
            GROUP BY tag 
            HAVING COUNT(*) >= 20   
        """
        )
        _TAG_NAMES.update(r[0] for r in rows)
    return _TAG_NAMES


def tag_for(word):
    for cand in (word, word[:-2], word[:-3] + "y", word[:-1]):
        if cand in _TAG_NAMES:
            return cand
    return None


def tag_score(tag, movie_id):
    if tag not in _TAGS:
        rows = fetch_all(
            "SELECT movie_id, score FROM movie_tag_scores WHERE tag = ?", (tag,)
        )
        _TAGS[tag] = {m: float(s) for m, s in rows}
        
    return _TAGS[tag].get(movie_id, 0.0)


def parse_tags(text, title):
    title_words = set(re.findall(r"[a-z0-9']+", title.lower()))
    tokens = re.findall(r"[a-z\-]+", text.lower())
    skip = FILLER | NEGATORS | STOP | title_words
    want, avoid = set(), set()
    for i, w in enumerate(tokens):
        if w in skip:
            continue
        
        tag = tag_for(w)
        if not tag:
            continue
        
        negated = any(x in NEGATORS for x in tokens[max(0, i - 2):i])
        (avoid if negated else want).add(tag)
        
    return want - avoid, avoid


def embedding_sim(a, b):
    va, vb = get_vec(a), get_vec(b)
    if va is None or vb is None:
        return None
    return max(0.0, float(va @ vb))


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
    skip = FILLER | set(KEYWORDS)
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

    tokens = re.findall(r"[a-z\-]+", text_lower)
    wanted, unwanted = set(), set()

    for i, word in enumerate(tokens):
        if word not in KEYWORDS:
            continue
        want, avoid = KEYWORDS[word]
        # a negator within the 2 previous words flips the meaning
        if any(w in NEGATORS for w in tokens[max(0, i - 2):i]):
            want, avoid = avoid, want
        wanted |= want
        unwanted |= avoid

    wanted -= unwanted
    return movie_id, wanted, unwanted


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


_INFO = {}

def recommend_details(text, n=16):
    movies = get_all_movies()
    
    if not _INFO:
        stats = get_movie_rating_stats()
        counts = dict(zip(stats["movieId"], stats["rating_count"]))
        avgs = dict(zip(stats["movieId"], stats["average_rating"]))
        
        for mid, title, genres, tmdb_id in movies:
            year = re.search(r"\((\d{4})\)\s*$", title)
            _INFO[title] = {
                "title": re.sub(r"\s*\(\d{4}\)\s*$", "", title),
                "year": year.group(1) if year else "",
                "genres": genres,
                "tmdb_id": tmdb_id,
                "rating": round(float(avgs.get(mid, 0)), 2),
                "count": int(counts.get(mid, 0))
            }
            
    titles = recommend_from_text(text, n)
    return [_INFO[t] for t in titles if t in _INFO]

def genre_fallback(movie_id, wanted, unwanted, info, total, avg, n):
    # used when almost nobody rated the movie: rank by genre overlap + quality
    base = set(info[movie_id][2].split("|")) | wanted
    scored = []
    for mid, (_, title, genres, _) in info.items():
        if mid == movie_id or total.get(mid, 0) < 50:
            continue
        g = set(genres.split("|"))
        if unwanted & g:
            continue
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

    movie_id, wanted, unwanted = parse_query(text, movies, total)
    if movie_id is None:
        return []

    info = {m[0]: m for m in movies}
    print(f"Matched movie: {info[movie_id][1]}  |  wanted: {wanted or 'none'}  |  unwanted: {unwanted or 'none'}")
    
    load_tags()
    want_tags, avoid_tags = parse_tags(text, info[movie_id][1])
    print(f"Tags wanted: {want_tags or 'none'} | Tags avoided: {avoid_tags or 'none'}")

    rows = get_fans(movie_id)

    candidates = []
    for mid, fans in rows:
        if mid not in info:
            continue

        genres = set(info[mid][2].split("|"))
        match = 1.0 if (wanted & genres) else 0.0
        bad = 1.0 if (unwanted & genres) else 0.0
        cf = fans / math.sqrt(total.get(mid, 1))
        c = total.get(mid, 0)
        a = float(avg.get(mid, 3.5))
        wr = (c / (c + 100)) * a + (100 / (c + 100)) * 3.5
        
        tb = sum(tag_score(t, mid) for t in want_tags) / max(1, len(want_tags))
        tp = sum(tag_score(t, mid) for t in avoid_tags) / max(1, len(avoid_tags))
        candidates.append((mid, cf, wr, match, bad, tb, tp))

    if not candidates:
        return genre_fallback(movie_id, wanted, unwanted, info, total, avg, n)

    if wanted:
        matched = [c for c in candidates if c[3]]
        if len(matched) >= n:
            candidates = matched

    if unwanted:
        clean = [c for c in candidates if not c[4]]
        if len(clean) >= n:
            candidates = clean

    max_cf = max(c[1] for c in candidates)
    scored = []
    for mid, cf, wr, match, bad, tb, tp in candidates:
        sim = embedding_sim(movie_id, mid)
        if sim is None:
            score = 0.6 * (cf / max_cf) + 0.25 * match + 0.15 * (wr / 5)
        else:
            score = 0.45 * (cf / max_cf) + 0.25 * sim + 0.2 * match + 0.1 * (wr / 5)
        score -= 0.4 * bad
        score += 2.0 * tb - 2.0 * tp
        scored.append((score, info[mid][1]))

    scored.sort(reverse=True)
    results = [title for _, title in scored[:n]]

    if len(results) < n:
        extra = genre_fallback(movie_id, wanted, unwanted, info, total, avg, n * 2)
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