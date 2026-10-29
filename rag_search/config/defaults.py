"""Search defaults and project-relative paths."""

import os
from pathlib import Path

BM25_B = 0.75
BM25_K1 = 1.5
DEFAULT_ALPHA = 0.5
DEFAULT_K = 60
DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP = 0
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DOCUMENT_PREVIEW_LENGTH = 100
SCORE_PRECISION = 3
SEARCH_MULTIPLIER = 5
RETRIEVAL_MULTIPLIER = 500
MAX_SEARCH_LIMIT = 100

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")
MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
