import polars as pl
from database.database import connect

df = (
    pl.read_csv("database/tags.csv", infer_schema_length=0)   # read everything as text
    .with_columns(
        pl.col("userId").cast(pl.Int64),
        pl.col("movieId").cast(pl.Int64),
        pl.col("tag").str.to_lowercase().str.strip_chars(),
        pl.col("timestamp").cast(pl.Int64),
    )
    .filter(pl.col("tag").str.len_chars().is_between(3, 100))
    .select("userId", "movieId", "tag", "timestamp")
)

conn = connect()
cur = conn.cursor()
rows = df.rows()
for i in range(0, len(rows), 20000):
    cur.executemany(
        "INSERT INTO tags (user_id, movie_id, tag, timestamp) VALUES (?, ?, ?, ?)",
        rows[i:i + 20000],
    )
    conn.commit()
    print(f"{min(i + 20000, len(rows)):,} / {len(rows):,}")
cur.close()
conn.close()