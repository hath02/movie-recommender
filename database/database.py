import mariadb

def connect():
    return mariadb.connect(
        host="localhost",
        port=3307,
        user="root",
        password="9291",
        database="movie_recommender"
    )
    
def fetch_all(query, params=None):
    conn = connect()
    cursor = conn.cursor()
    
    cursor.execute(query, params or ())
    rows = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return rows