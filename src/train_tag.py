import torch
import torch.nn as nn
import torch.nn.functional as F
from database.database import fetch_all

torch.manual_seed(42)
N_TAGS = 1000

# 1) MF embeddings (the targets)
ck = torch.load("models/movy_mf.pt", map_location="cpu")
emb = F.normalize(ck["model_state_dict"]["movie_embedding.weight"], dim=1)
m2i = ck["movie_to_index"]

# 2) tag matrix (the inputs): top 1000 tags
tags = [r[0] for r in fetch_all(
    f"SELECT tag FROM movie_tag_scores GROUP BY tag ORDER BY COUNT(*) DESC LIMIT {N_TAGS}"
)]
tag_idx = {t: i for i, t in enumerate(tags)}

rows = [r for r in fetch_all("SELECT movie_id, tag, score FROM movie_tag_scores")
        if r[1] in tag_idx]
movie_ids = sorted({r[0] for r in rows})
row_of = {m: i for i, m in enumerate(movie_ids)}

X = torch.zeros(len(movie_ids), N_TAGS)
for m, t, s in rows:
    X[row_of[m], tag_idx[t]] = float(s)
X = F.normalize(X, dim=1)
print(f"{len(movie_ids):,} movies with tags")

# 3) training movies: have an MF embedding and 500+ ratings (reliable target)
counts = dict(fetch_all("SELECT movie_id, rating_count FROM movie_stats"))
train_movies = [m for m in movie_ids if m in m2i and counts.get(m, 0) >= 500]
perm = torch.randperm(len(train_movies)).tolist()
cut = int(len(perm) * 0.9)
tr = [train_movies[i] for i in perm[:cut]]
va = [train_movies[i] for i in perm[cut:]]
print(f"train {len(tr):,} | validation {len(va):,}")

def get(movies):
    return X[[row_of[m] for m in movies]], emb[[m2i[m] for m in movies]]

Xtr, Ytr = get(tr)
Xva, Yva = get(va)

# baseline: always predict the average embedding
mean_vec = F.normalize(Ytr.mean(dim=0, keepdim=True), dim=1)
base = (Yva @ mean_vec.T).mean().item()
print(f"Baseline validation cosine: {base:.3f}")

# 4) the network
net = nn.Sequential(
    nn.Linear(N_TAGS, 256), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(256, 32),
)
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
best = -1.0

for epoch in range(1, 301):
    net.train()
    order = torch.randperm(len(Xtr))
    for i in range(0, len(order), 256):
        idx = order[i:i + 256]
        loss = 1 - F.cosine_similarity(net(Xtr[idx]), Ytr[idx]).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    if epoch % 10 == 0:
        net.eval()
        with torch.no_grad():
            val = F.cosine_similarity(net(Xva), Yva).mean().item()
        print(f"Epoch {epoch} - validation cosine {val:.3f}")
        if val > best:
            best = val
            torch.save(net.state_dict(), "models/tag_net.pt")

# 5) predict an embedding for every movie that has tags
net.load_state_dict(torch.load("models/tag_net.pt"))
net.eval()
with torch.no_grad():
    pred = F.normalize(net(X), dim=1)
torch.save({"movie_ids": movie_ids, "emb": pred}, "models/tag_emb.pt")
print(f"Best validation cosine: {best:.3f} (baseline {base:.3f})")