"""Keyword command implementations; argument parsing stays in cli/."""

from rag_search.config.defaults import DEFAULT_SEARCH_LIMIT, BM25_K1, BM25_B
from rag_search.search.keyword_search import InvertedIndex
from rag_search.search.tokenization import tokenize_text, has_matching_token, _single_token
from rag_search.sources.movie_loader import load_movies


@DeprecationWarning
def search_command_old(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    """
    Search for movies whose titles contain any token from the query (case-insensitive, ignoring punctuation).
    Returns up to 'limit' matching movies as a dict of {id: movie}.
    """
    movies = load_movies()
    results = []

    # Preprocess query once (outside loop)
    query_tokens = tokenize_text(query)

    for movie in movies:
        # Preprocess title once per movie
        title_tokens = tokenize_text(movie["title"])

        # Check for intersection (any query token matches any title token as substring)
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break  # Early exit to avoid unnecessary checks

    return results


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies_idx = InvertedIndex()
    movies_idx.load()

    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = movies_idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = movies_idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

    return results


def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies_idx = InvertedIndex()
    movies_idx.load()
    return movies_idx.bm25_search(query, limit)


def tf_commmand(doc_id: int, term: str) -> int:
    movies_idx = InvertedIndex()
    movies_idx.load()
    token = _single_token(term)
    return movies_idx.get_tf(doc_id, token)


def idf_command(term: str) -> float:
    movies_idx = InvertedIndex()
    movies_idx.load()
    token = _single_token(term)
    return movies_idx.get_idf(token)


def bm25_idf_command(term: str) -> float:
    movies_idx = InvertedIndex()
    movies_idx.load()
    token = _single_token(term)
    return movies_idx.get_bm25_idf(token)


def tfidf_command(doc_id: int, term: str) -> float:
    movies_idx = InvertedIndex()
    movies_idx.load()
    token = _single_token(term)
    return movies_idx.get_tfidf(doc_id, token)


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    token = _single_token(term)
    return idx.get_bm25_tf(doc_id, token, k1, b)


def build_command(verbose=False):


    midx = InvertedIndex(verbose=verbose)
    midx.build()
    midx.save()
