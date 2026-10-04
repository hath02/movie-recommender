import polars as pl
from database.database import connect

def read_dat(path, names):
    return pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        quote_char=None,            # tags may contain quotes
        infer_schema_length=0,      # read everything as text
        columns=list(range(len(names))),
        new_columns=names,
        encoding="utf8-lossy",
    )

scores = read_dat("database/tag_relevance.dat", ["movieId", "tagId", "relevance"]).with_columns(
    pl.col("movieId").cast(pl.Int64),
    pl.col("tagId").cast(pl.Int64),
    pl.col("relevance").cast(pl.Float64),
)
tags = read_dat("database/tags.dat", ["tagId", "tag"]).with_columns(
    pl.col("tagId").cast(pl.Int64),
    pl.col("tag").str.to_lowercase().str.strip_chars(),
)

df = (
    scores.filter(pl.col("relevance") >= 0.3)      # drop weak scores
    .join(tags, on="tagId")
    .select("movieId", "tag", "relevance")
)
rows = df.rows()
print(f"{len(rows):,} rows to insert")

conn = connect()
cur = conn.cursor()
for i in range(0, len(rows), 20000):
    cur.executemany(
        "INSERT INTO movie_tag_scores_g (movie_id, tag, score) VALUES (?, ?, ?)",
        rows[i:i + 20000],
    )
    conn.commit()
    print(f"{min(i + 20000, len(rows)):,} / {len(rows):,}")
cur.close()
conn.close()