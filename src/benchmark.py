import time

from src.recommendation import (
    recommend_for_user,
    get_user_rated_movies
)


user_id = 32
movie_title = "Seven"

rated_movies = get_user_rated_movies(user_id)


# Warm-up
print("Warming up...")

recommend_for_user(
    movie_title,
    user_id,
    n=10,
    exclude_movies=rated_movies
)

print("Warm-up complete.\n")


# Benchmark
times = []

runs = 5

for i in range(runs):
    start = time.perf_counter()

    recommend_for_user(
        movie_title,
        user_id,
        n=10,
        exclude_movies=rated_movies
    )

    elapsed = time.perf_counter() - start
    times.append(elapsed)

    print(f"Run {i + 1}: {elapsed:.4f}s")


print()
print(f"Average: {sum(times) / len(times):.4f}s")
print(f"Fastest: {min(times):.4f}s")
print(f"Slowest: {max(times):.4f}s")