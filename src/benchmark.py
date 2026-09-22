import time
from src.recommendation import recommend_for_user

def benchmark_recommendation(movie_title, user_id, runs=5):
    times = []
    
    for _ in range(runs):
        start = time.perf_counter()
        
        recommend_for_user(
            movie_title,
            user_id,
            n=10
        )
        
        end = time.perf_counter()
        
        times.append(end - start)
        
    print(f"Runs: {runs}")
    print(f"Average: {sum(time) / len(times):.4f}s")
    print(f"Fastest: {min(time):.4f}s")
    print(f"Slowest: {max(time):.4f}s")
    
benchmark_recommendation(
    movie_title="Seven",
    user_id=26,
    runs=5
)