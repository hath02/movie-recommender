from database.database import fetch_all

movies = fetch_all("""
    SELECT movie_id, title, genres
    FROM movies
""")

print(movies[:5])