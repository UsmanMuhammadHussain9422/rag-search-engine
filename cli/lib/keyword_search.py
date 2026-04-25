from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        if str(movie["title"]).__contains__(query):
            results.append(movie) 
        if len(results) == limit:
            break
    return results