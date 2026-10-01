CREATE TABLE movie_stats AS
SELECT movie_id,
       COUNT(*) AS rating_count,
       CAST(AVG(rating) AS DOUBLE) AS average_rating
FROM ratings
GROUP BY movie_id;