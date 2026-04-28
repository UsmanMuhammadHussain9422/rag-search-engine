from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies
import string

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict[int, dict]:
    """
    Search for movies whose titles contain any token from the query (case-insensitive, ignoring punctuation).
    Returns up to 'limit' matching movies as a dict of {id: movie}.
    """
    movies = load_movies()
    results: dict[int, dict] = {}
    
    # Preprocess query once (outside loop)
    query_tokens = set(preprocess_text(query))
    
    for movie in movies:
        # Preprocess title once per movie
        title_tokens = set(preprocess_text(movie["title"]))
        
        # Check for intersection (any query token matches any title token as substring)
        if any(q_token in t_token for q_token in query_tokens for t_token in title_tokens):
            results[movie['id']] = movie
            if len(results) == limit:
                break  # Early exit to avoid unnecessary checks
    
    return results

def preprocess_text(text: str) -> list[str]:
    """Preprocess text: remove punctuation, lowercase, tokenize."""
    processed_text = remove_punctuation(text).lower()
    return tokenize_text(processed_text)

def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))

def tokenize_text(text: str) -> list[str]:
    return text.split()