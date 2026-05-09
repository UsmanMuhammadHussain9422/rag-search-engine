import json
import os

DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

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