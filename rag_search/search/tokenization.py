"""Keyword tokenization and a stopword list cached for the process lifetime."""

import string
from nltk.stem import PorterStemmer

from rag_search.config.defaults import STOPWORDS_PATH

STOP_WORDS: list[str] | None = None


def load_stopwords() -> list[str]:

    global STOP_WORDS

    if STOP_WORDS is None:
        with open(STOPWORDS_PATH, "r") as f:
            STOP_WORDS = f.read().splitlines()
    return STOP_WORDS


def has_matching_token(query_tokens, title_tokens):
    return any(q_token in t_token for q_token in query_tokens for t_token in title_tokens)


def preprocess_text(text: str) -> str:
    """Preprocess text: remove punctuation, lowercase, tokenize."""
    return remove_punctuation(text).lower()


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    filtered_words = remove_stopwords(tokens)
    stemmed_words = stemmer(filtered_words)
    return stemmed_words


def _single_token(term: str) -> str:
    """Tokenize a CLI-supplied term and ensure exactly one usable token remains."""
    tokens = tokenize_text(term)
    if len(tokens) == 0:
        raise ValueError(f"term '{term}' produced no tokens (likely a stopword)")
    if len(tokens) > 1:
        raise ValueError(f"term '{term}' must be a single token, got {len(tokens)}")
    return tokens[0]


def remove_stopwords(words: list[str]) -> list[str]:
    stopwords = set(load_stopwords())  # Convert to set for fast lookup; cache if possible
    return [word for word in words if word not in stopwords]


def stemmer(texts: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(text) for text in texts]
