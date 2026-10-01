import torch

from src.model import MatrixFactorization


num_users = 100
num_movies = 500

model = MatrixFactorization(
    num_users=num_users,
    num_movies=num_movies,
    embedding_dim=32
)


user_ids = torch.tensor(
    [0, 1, 2, 3],
    dtype=torch.long
)

movie_ids = torch.tensor(
    [10, 20, 30, 40],
    dtype=torch.long
)


predictions = model(
    user_ids,
    movie_ids
)


print("Predictions:")
print(predictions)

print()
print("Shape:")
print(predictions.shape)