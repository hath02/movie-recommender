import json
from collections import defaultdict
from database.database import fetch_all

genre_of = {m: set(g.split("|")) for m, g in fetch_all(
    "SELECT movie_id, genres FROM movies "
    "WHERE movie_id IN (SELECT DISTINCT movie_id FROM movie_tag_scores_g)")}
n_all = len(genre_of)

n_genre = defaultdict(int)
for gs in genre_of.values():
    for g in gs:
        n_genre[g] += 1

tag_movies = defaultdict(set)
for m, t in fetch_all("SELECT movie_id, tag FROM movie_tag_scores_g"):
    if m in genre_of:
        tag_movies[t].add(m)

lexicon = {}
for t, ms in tag_movies.items():
    if len(ms) < 30:
        continue

    base = len(ms) / n_all
    boost, avoid = [], []
    for g, n in n_genre.items():
        if n < 100 or g == "(no genres listed)":
            continue
        lift = (sum(1 for m in ms if g in genre_of[m]) / n) / base
        if lift >= 2.0:
            boost.append((lift, g))
        elif lift <= 0.5:
            avoid.append((lift, g))

    boost.sort(reverse=True)          # strongest association first
    avoid.sort()                      # weakest (most avoided) first
    lexicon[t] = {
        "boost": [g for _, g in boost],
        "avoid": [g for _, g in avoid],
    }

with open("models/tag_genre_lexicon.json", "w") as f:
    json.dump(lexicon, f, indent=1)

print(len(lexicon), "tags learned")
for t in ("dark", "funny", "violent", "sad"):
    print(t, lexicon.get(t))