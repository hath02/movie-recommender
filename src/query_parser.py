import re
import math
import json
import os
import urllib.request
import torch
import torch.nn.functional as F
import difflib
from src.recommendation import fetch_all, get_all_movies, get_movie_rating_stats


try:
    LEXICON = json.load(open("models/tag_genre_lexicon.json"))
except FileNotFoundError:
    LEXICON = {}
    
    
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
# this is old and not used anymore, but kept for backwards compatibility with old queries
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

POP_RE = re.compile(
    r"\b(popular\w*|famous|blockbusters?|mainstream|trending|well-known|"
    r"obscure|underrated|hidden|gems?|unknown|lesser-known|unpopular|indie|niche)\b"
)
OBSCURE_WORDS = {"obscure", "underrated", "hidden", "gem", "gems", "unknown",
                 "lesser-known", "unpopular", "indie", "niche"}

STOP = {"the", "of", "in", "for", "on", "it", "is", "that", "this", "are",
        "can", "at", "as", "be", "just", "very", "too", "bit", "little",
        "any", "then", "than", "from", "have", "has", "was", "were"}

_MF = {"emb": None, "m2i": None}

_TE = {"emb": None, "m2i": None}

_TAG_NAMES = set()
_TAGS = {}  # tags -> {movie_id: score}, filled lazily

_PEOPLE = {}  # movie_id -> [(person_id, name, role)], filled lazily
_PERSON_MOVIES = {}  # person_id -> [(movie_id, role)], filled lazily
_LAST = {} # last name -> person_id, filled lazily
_PERSON_ROLE_MOVIES = {}   # (person_id, role) -> set of movie_ids

_TITLE_SET = set()  # all movie titles, filled lazily

DIRECTOR_CUES = {"directed", "director", "helmed",
                 "directed by", "by", "filmed by",
                 "helmed by", "directors", "helmers",
                 "helmer", "filmmaker", "filmmakers",
                 "filmmaking", "filmmaker's", "filmmakers'", "helmed by"}

ACTOR_CUES = {"starring", "starred", "stars", 
              "cast", "featuring", "featuring", 
              "feats", "with", "acted",
              "actor", "actress", "performer", "performers",
              "played", "plays", "portrayed", "portrays"}


PERSON_CACHE = "models/person_cache.json"
try:
    with open(PERSON_CACHE) as f:
        _PC = json.load(f)
except Exception:
    _PC = {}

_TMDB2MOVIE = {}
_EXTENDED = set()

    
    
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


def strip_pop(text_lower):
    return POP_RE.sub(" ", text_lower)


def parse_popularity(text):
    """+1 = wants popular movies, -1 = wants obscure ones, 0 = no preference."""
    tokens = re.findall(r"[a-z\-]+", text.lower())
    for i, w in enumerate(tokens):
        if not POP_RE.fullmatch(w):
            continue
        direction = -1 if w in OBSCURE_WORDS else 1
        if any(x in NEGATORS for x in tokens[max(0, i - 2):i]):
            direction = -direction          # "less popular" -> obscure
        return direction
    return 0


def extend_person(pid):
    if pid in _EXTENDED:
        return
    key = str(pid)

    if key not in _PC:
        api_key = os.environ.get("TMDB_API_KEY")
        if not api_key:
            return
        url = f"https://api.themoviedb.org/3/person/{pid}/movie_credits?api_key={api_key}"
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                d = json.load(r)
        except Exception as e:
            print("person fetch failed:", e)
            return
        _PC[key] = {
            "director": [m["id"] for m in d.get("crew", []) if m.get("job") == "Director"],
            "actor": [m["id"] for m in d.get("cast", []) if m.get("order", 99) < 8],
        }
        with open(PERSON_CACHE, "w") as f:
            json.dump(_PC, f)

    if not _TMDB2MOVIE:
        for mid, tmdb in fetch_all("SELECT movie_id, tmdb_id FROM links WHERE tmdb_id IS NOT NULL"):
            _TMDB2MOVIE[int(float(tmdb))] = mid

    for role, tmdb_ids in _PC[key].items():
        for t in tmdb_ids:
            mid = _TMDB2MOVIE.get(t)
            if mid:
                _PERSON_MOVIES.setdefault(pid, set()).add(mid)
                _PERSON_ROLE_MOVIES.setdefault((pid, role), set()).add(mid)

    _EXTENDED.add(pid)
    
    
def load_people():
    if _PEOPLE:
        return
    for pid, name in fetch_all("SELECT person_id, name FROM people"):
        _PEOPLE[name.lower()] = pid
    for pid, mid, role in fetch_all("SELECT person_id, movie_id, role FROM movie_people"):
        _PERSON_MOVIES.setdefault(pid, set()).add(mid)
        _PERSON_ROLE_MOVIES.setdefault((pid, role), set()).add(mid)
    _TITLE_SET.update(clean_title(m[1]) for m in get_all_movies())


def find_people(text_lower):
    """Returns ([(person_id, role_or_None, negated), ...], text with names/cues removed)."""
    tokens = re.findall(r"[a-z0-9\-']+", text_lower)
    used, found = set(), []
    skip = FILLER | STOP | NEGATORS | DIRECTOR_CUES | ACTOR_CUES

    def add(pid, i, size):
        role, negated = None, False
        for j in range(i - 1, max(-1, i - 4), -1):      # up to 3 words before
            if j in used or tokens[j] in {"but", "and"}:
                break                                   # don't cross into another clause
            if tokens[j] in NEGATORS:
                negated = True
                used.add(j)
            elif tokens[j] in DIRECTOR_CUES:
                role = "director"
                used.add(j)
            elif tokens[j] in ACTOR_CUES:
                role = "actor"
                used.add(j)
        found.append((pid, role, negated))
        used.update(range(i, i + size))

    # pass 1: exact names
    for size in (4, 3, 2):
        for i in range(len(tokens) - size + 1):
            if any(j in used for j in range(i, i + size)):
                continue
            phrase = " ".join(tokens[i:i + size])
            if phrase in _PEOPLE and phrase not in _TITLE_SET:
                add(_PEOPLE[phrase], i, size)

    # pass 2: fuzzy names ("cillian murp", "cristopher nolan")
    for size in (3, 2):
        for i in range(len(tokens) - size + 1):
            if any(j in used for j in range(i, i + size)):
                continue
            if tokens[i] in skip or tokens[i + size - 1] in skip:
                continue
            phrase = " ".join(tokens[i:i + size])
            if len(phrase) < 6 or phrase in _TITLE_SET:
                continue
            m = difflib.get_close_matches(phrase, _PEOPLE, n=1, cutoff=0.88)
            if m:
                add(_PEOPLE[m[0]], i, size)

    cleaned = " ".join(t for j, t in enumerate(tokens) if j not in used)
    return found, cleaned
                
                            
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
            FROM movie_tag_scores_g
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
            "SELECT movie_id, score FROM movie_tag_scores_g WHERE tag = ?", (tag,)
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
    skip = FILLER | NEGATORS | STOP | set(KEYWORDS)
    best_ratio, best_id = 0.0, None

    for size in range(1, 5):
        for i in range(len(tokens) - size + 1):
            gram = tokens[i:i + size]
            if gram[0] in skip or gram[-1] in skip:
                continue
            phrase = " ".join(gram)
            if len(phrase) < 4:
                continue
            for name in difflib.get_close_matches(phrase, titles, n=1, cutoff=0.85):
                ratio = difflib.SequenceMatcher(None, phrase, name).ratio()
                if ratio > best_ratio:
                    best_ratio, best_id = ratio, titles[name]

    return best_id


def parse_query(text, movies, total):
    load_tags()
    text_lower = text.lower()
    movie_id = find_movie(strip_pop(text_lower), movies, total)

    tokens = re.findall(r"[a-z\-]+", text_lower)
    wanted, unwanted = set(), set()

    for i, word in enumerate(tokens):
        tag = None
        if word not in GENRE_KEYWORDS and word not in FILLER | STOP | NEGATORS and len(word) > 3:
            tag = tag_for(word)

        if word in GENRE_KEYWORDS:                 # explicit genre: keep the rule
            want, avoid = KEYWORDS[word]
        elif tag in LEXICON:                       # learned from data
            want, avoid = set(LEXICON[tag]["boost"]), set(LEXICON[tag]["avoid"])
        elif word in KEYWORDS:                     # old MOODS as fallback
            want, avoid = KEYWORDS[word]
        else:
            continue

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


def browse(wanted, unwanted, pop, info, total, avg, n, allowed=None, excluded=None):
    # used when the user gives no reference movie, e.g. "less popular comedy"
    scored = []
    for mid, (_, title, genres, _) in info.items():
        if allowed is not None and mid not in allowed:
            continue
        
        if excluded and mid in excluded:
            continue
        
        c = total.get(mid, 0)
        if c < 50:
            continue
        if pop < 0 and c > 3000:      # "obscure" = fewer than 3000 ratings
            continue
        g = set(genres.split("|"))
        if unwanted & g:
            continue
        if wanted and not (wanted & g):
            continue
        a = float(avg.get(mid, 3.5))
        wr = (c / (c + 100)) * a + (100 / (c + 100)) * 3.5
        norm = min(1.0, math.log10(c + 1) / 6)       # 0 = unknown, 1 = hugely popular
        p = norm if pop > 0 else ((1 - norm) if pop < 0 else 0.0)
        scored.append((0.7 * (wr / 5) + 0.3 * p, title))
    scored.sort(reverse=True)
    return [t for _, t in scored[:n]]


def recommend_from_text(text, n=16):
    movies = get_all_movies()
    load_mf()
    stats = get_movie_rating_stats()
    total = dict(zip(stats["movieId"], stats["rating_count"]))
    avg = dict(zip(stats["movieId"], stats["average_rating"]))

    load_people()
    people, cleaned = find_people(text.lower())
    text = cleaned if people else text.lower()

    pos_sets, excluded = [], set()
    for pid, role, negated in people:
        extend_person(pid)
        s = (_PERSON_ROLE_MOVIES.get((pid, role), set()) if role else set()) \
            or _PERSON_MOVIES.get(pid, set())
        if negated:
            excluded |= s
        else:
            pos_sets.append(s)

    person_movies = set()
    if pos_sets:
        person_movies = set.intersection(*pos_sets) or set.union(*pos_sets)
    person_movies -= excluded
        
    movie_id, wanted, unwanted = parse_query(text, movies, total)
    pop = parse_popularity(text)
    info = {m[0]: m for m in movies}
    
    if movie_id is None:
        if wanted or unwanted or pop or person_movies or excluded:
            print(f"No reference movie | wanted: {wanted or 'none'} | popularity: {pop} | people: {len(people)} | excluded: {len(excluded)}")
            allowed = person_movies if person_movies else None
            return browse(wanted, unwanted, pop, info, total, avg, n, allowed, excluded)
        return []
    
    print(f"Matched movie: {info[movie_id][1]}  |  wanted: {wanted or 'none'}  |  unwanted: {unwanted or 'none'}")
    
    load_tags()
    want_tags, avoid_tags = parse_tags(text, info[movie_id][1])
    print(f"Tags wanted: {want_tags or 'none'} | Tags avoided: {avoid_tags or 'none'}")

    rows = get_fans(movie_id)
    if person_movies:
        seen = {r[0] for r in rows}
        rows = list(rows) + [(m, 1) for m in person_movies if m not in seen and m != movie_id]

    candidates = []
    for mid, fans in rows:
        if mid not in info or mid in excluded:
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
        score += 0.4 * tb - 0.4 * tp
        norm = min(1.0, math.log10(total.get(mid, 1) + 1) / 6)       # 0 = unknown, 1 = hugely popular
        if pop > 0:
            score += 0.2 * norm
        elif pop < 0:
            score += 0.2 * (1 - norm)
        if mid in person_movies:
            score += 0.3
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