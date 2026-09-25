import time
from src.recommendation import recommend_for_user
from src.evaluation import train_test_split, precision_at_k

user_id = 26
movie_title = "Seven"

train_movies, test_movies = train_test_split(user_id)

recommendations = recommend_for_user(
    movie_title,
    user_id,
    n=10,
    exclude_movies=train_movies
)

precision = precision_at_k(
    recommendations,
    test_movies,
    10
)



print("Recommendations:")
print(
    recommendations.select(
        ["movieId", "title", "final_score"]
    )
)

print("\nTraining movies:")
print(train_movies)

print("\nTest / relevant movies:")
print(test_movies)

print(f"\nPrecision@10: {precision:.2f}")