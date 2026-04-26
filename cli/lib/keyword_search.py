from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies
import string

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        preprocessed_query = preprocess_text(query)[0]
        preprocessed_title = preprocess_text(movie["title"])[0]
        if preprocessed_query in preprocessed_title:
            results.append(movie)
        if len(results) == limit:
            break
    return results

def preprocess_text(text: str) -> list[str]:
    preprocess_text = remove_punctuation(text)
    preprocess_text = preprocess_text.lower()
    return [preprocess_text]

def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))