import polars as pl
from database.database import fetch_all

def genre_similarity(genres1, genres2):
    set1 = set(genres1)
    set2 = set(genres2)

    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union

# Get similar movies
def get_content_recommendations(movie_title, n=10):
    # Find the target movie
    target_rows = fetch_all("""
        SELECT movie_id, title, genres
        from movies 
        WHERE title LIKE ?
        LIMIT 1
    """, (f"%{movie_title}%",))

    if not target_rows:
        return pl.DataFrame({
            "title": ["MOVIE NOT FOUND"]
        })

    target_movie = target_rows[0]      
    
    target_movie_id = target_movie[0]
    target_genres = target_movie[2].split("|")

    # Get all similar movies
    movie_rows = fetch_all("""
        SELECT movie_id, title, genres
        FROM movies
    """)

    results = []
    
    for movie_id, title, genres in movie_rows:
        if movie_id == target_movie_id:
            continue
        
        genre_list = genres.split("|")
        
        similarity = genre_similarity(
            target_genres,
            genre_list
        )

        results.append({
            "movieId": movie_id,
            "title": title,
            "genres": genres,
            "similarity": similarity
        })

    return (
        pl.DataFrame(results)
        .sort("similarity", descending=True)
        .head(n)
    )

# Get rating scores
def get_movie_rating_stats():
    rows = fetch_all("""
        SELECT 
            movie_id,
            COUNT(*) AS rating_count,
            AVG(rating) AS average_rating
        FROM ratings
        GROUP BY movie_id
    """)
    
    return pl.DataFrame(
        rows,
        schema=[
            "movieId",
            "rating_count",
            "average_rating"
        ],
        orient="row"
    )

# Genre scores for getting similar movies
def get_genre_scores(user_id):
    rows = fetch_all("""
        SELECT m.genres
        FROM ratings r
        JOIN movies m
            ON r.movie_id = m.movie_id
        WHERE r.user_id = ?
            AND r.rating >= 4.0
    """, (user_id,)) 

    genre_scores = {}
    
    for (genres,) in rows:
        for genre in genres.split("|"):
            genre_scores[genre] = genre_scores.get(genre, 0) + 1
            
    return genre_scores
    
# Compare to personal rated movies
def personalized_score(genres, genre_scores):
    if not genre_scores:
        return 0.0

    genre_list = genres.split("|")

    if not genre_list:
        return 0.0

    max_genre_score = max(genre_scores.values())

    if max_genre_score == 0:
        return 0.0

    score = sum(
        genre_scores.get(genre, 0)
        for genre in genre_list
    ) / len(genre_list)

    return score / max_genre_score

def get_user_rated_movies(user_id):
    rows = fetch_all("""
        SELECT movie_id
        FROM ratings
        WHERE user_id = ?
    """, (user_id,))

    return {row[0] for row in rows}

def recommend_for_user(movie_title, user_id, n=10, exclude_movies=None):
    
    exclude_movies = set(exclude_movies or [])
    
    n = min(n, 10)
    
    # Get this user's genre preferences
    genre_scores = get_genre_scores(user_id)    

    # Get content-based recommendations
    recommendations = get_content_recommendations(
        movie_title,
        n=1000
    )
    
    recommendations = recommendations.filter(
        ~pl.col("movieId").is_in(exclude_movies)
    )

    # Calculate personalized score
    recommendations = recommendations.with_columns(
        pl.col("genres")
        .map_elements(
            lambda genres: personalized_score(
                genres,
                genre_scores
            ),
            return_dtype=pl.Float64
        )
        .alias("personalized_score")    
    )

    # Calculate final hybrid score
    recommendations = recommendations.with_columns(
        (
            0.5 * pl.col("similarity")
            + 0.5 * pl.col("personalized_score")
        ).alias("final_score")
    )

    # Add rating statistics
    recommendations = recommendations.join(
        get_movie_rating_stats(),
        on="movieId",
        how="left"
    )
    
    # Sort by final score
    return (
        recommendations
        .sort("final_score", descending=True)
        .head(n)
    )