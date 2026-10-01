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

def fetch_batches(query, params=None, batch_size=10000):
    conn = connect()
    cursor =conn.cursor()
    
    try:
        cursor.execute(query, params or ())
        
        while True:
            rows = cursor.fetchmany(batch_size)
            
            if not rows:
                break
            
            yield rows
            
    finally:
        cursor.close()
        conn.close()