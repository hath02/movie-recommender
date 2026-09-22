from database.database import fetch_all

# Get user's highly-rated movies
def get_relevant_movies(user_id):
    rows = fetch_all("""
        SELECT movie_id
        FROM ratings
        WHERE user_id = ?
          AND rating >= 4.0
    """, (user_id,))

    return {row[0] for row in rows}


# Get user's rated movies
def get_rated_movies(user_id):
    rows = fetch_all("""
        SELECT movie_id
        FROM ratings
        WHERE user_id = ?
    """, (user_id,))

    return {row[0] for row in rows}


# Split relevant movies into training and test sets
def train_test_split(user_id, test_ratio=0.2):
    relevant_movies = sorted(get_relevant_movies(user_id))

    split_index = int(len(relevant_movies) * (1 - test_ratio))

    train_movies = set(relevant_movies[:split_index])
    test_movies = set(relevant_movies[split_index:])

    return train_movies, test_movies


# Calculate Precision@K
def precision_at_k(recommended, relevant, k):
    top_k = recommended.head(k)

    recommendation_ids = top_k["movieId"].to_list()

    if not recommendation_ids:
        return 0.0

    relevant_count = sum(
        movie_id in relevant
        for movie_id in recommendation_ids
    )

    return relevant_count / len(recommendation_ids)