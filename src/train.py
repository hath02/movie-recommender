import os
import torch
import torch.nn as nn
from database.database import fetch_batches
from src.model import MatrixFactorization

EMBEDDING_DIM = 32
BATCH_SIZE = 4096
EPOCHS = 10
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-5
MODEL_PATH = os.path.join("models", "movy_mf.pt")

torch.manual_seed(42)


def load_ratings():
    users, movies, ratings = [], [], []
    for batch in fetch_batches(
        "SELECT user_id, movie_id, rating FROM ratings WHERE MOD(user_id, 10) = 0"
    ):
        for u, m, r in batch:
            users.append(u)
            movies.append(m)
            ratings.append(float(r))
    print(f"Loaded {len(ratings):,} ratings")
    return users, movies, ratings


def evaluate(model, u, m, r):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(r), 16384):
            pred = model(u[i:i + 16384], m[i:i + 16384])
            total += ((pred - r[i:i + 16384]) ** 2).sum().item()
    return (total / len(r)) ** 0.5


def train():
    users, movies, ratings = load_ratings()

    user_to_index = {u: i for i, u in enumerate(sorted(set(users)))}
    movie_to_index = {m: i for i, m in enumerate(sorted(set(movies)))}

    u = torch.tensor([user_to_index[x] for x in users], dtype=torch.long)
    m = torch.tensor([movie_to_index[x] for x in movies], dtype=torch.long)
    r = torch.tensor(ratings, dtype=torch.float32)
    mean_rating = r.mean().item()

    # 90% train / 10% validation
    perm = torch.randperm(len(r))
    cut = int(len(r) * 0.9)
    tr, va = perm[:cut], perm[cut:]
    tu, tm, trr = u[tr], m[tr], r[tr]
    vu, vm, vr = u[va], m[va], r[va]

    # baseline: always predict the average rating
    baseline = ((vr - mean_rating) ** 2).mean().item() ** 0.5
    print(f"Baseline RMSE (predict the mean): {baseline:.4f}")

    model = MatrixFactorization(len(user_to_index), len(movie_to_index), EMBEDDING_DIM)
    model.global_bias.data.fill_(mean_rating)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    os.makedirs("models", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(trr))
        total, count = 0.0, 0
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(tu[idx], tm[idx]), trr[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
            count += len(idx)

        train_rmse = (total / count) ** 0.5
        val_rmse = evaluate(model, vu, vm, vr)
        print(f"Epoch {epoch + 1}/{EPOCHS} - train RMSE {train_rmse:.4f} - val RMSE {val_rmse:.4f}")

        if val_rmse < best_val:
            best_val = val_rmse
            torch.save({
                "model_state_dict": model.state_dict(),
                "user_to_index": user_to_index,
                "movie_to_index": movie_to_index,
                "num_users": len(user_to_index),
                "num_movies": len(movie_to_index),
                "embedding_dim": EMBEDDING_DIM,
                "mean_rating": mean_rating,
            }, MODEL_PATH)
            print("  saved (best so far)")

    print(f"Best validation RMSE: {best_val:.4f}")


if __name__ == "__main__":
    train()