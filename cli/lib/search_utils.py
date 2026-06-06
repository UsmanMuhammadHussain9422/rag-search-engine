import json
import os
from typing import Any
from .custom_types import SearchResult

BM25_B = 0.75
BM25_K1 = 1.5
DEFAULT_ALPHA = 0.5
DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP = 0
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DOCUMENT_PREVIEW_LENGTH = 100
SCORE_PRECISION = 3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")
MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")

STOP_WORDS:list[str] = []

def smart_print(message, should_print=True):
    if should_print: 
        print(message)

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords() -> list[str]:

    global STOP_WORDS

    if STOP_WORDS is None or len(STOP_WORDS) == 0:
        with open(STOPWORDS_PATH, "r") as f:
            STOP_WORDS = f.read().splitlines()
    return STOP_WORDS

def format_search_result(doc_id: int, title: str, document: str, score: float, **metadata: Any) -> SearchResult:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document[:DOCUMENT_PREVIEW_LENGTH],
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata or {}
    }