"""Semantic command implementations; argument parsing stays in cli/."""

import numpy as np
from rag_search.config.defaults import (
    DEFAULT_SEARCH_LIMIT, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP,
    DEFAULT_CHUNK_OVERLAP, DEFAULT_SEMANTIC_CHUNK_SIZE,
)
from rag_search.search.semantic_search import (
    SemanticSearch, ChunkedSemanticSearch, get_text_chunks, get_semantic_text_chunks,
)
from rag_search.sources.movie_loader import load_movies


def verify_model() -> None:
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")
    return None


def embed_text(text: str):
    sentence_Search = SemanticSearch()
    embedding = sentence_Search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    documents: list[dict] = load_movies()
    search_instance = SemanticSearch()
    embeddings = search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query_text(query:str):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def search(query: str, limit: int):
    documents = load_movies()
    search_instance = SemanticSearch()
    search_instance.load_or_create_embeddings(documents)
    results = search_instance.search(query, limit)
    for index, res in enumerate(results):
        print(f"{index + 1}. {res['title']} (score: {res['score']:.4f})\n")


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
    """
    Splits the input text into fixed-size chunks and prints each chunk.

    Args:
        text (str): The text to be chunked.
        chunk_size (int): The number of words per chunk.
    """
    chunks = get_text_chunks(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def semantic_chunk_text(text: str, max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
        """
        Splits the input text into semantic(sentence) chunks and prints each chunk.

        Args:
            text (str): The text to be chunked.
            max_chunk_size (int): The number of sentences per chunk.
        """

        chunks = get_semantic_text_chunks(text, max_chunk_size, overlap)
        print(f"Semantically chunking {len(text)} characters")
        for i, chunk in enumerate(chunks):
            print(f"{i + 1}. {chunk}")


def embed_chunks() -> np.ndarray:
    documents = load_movies()
    search_engine = ChunkedSemanticSearch()
    return search_engine.load_or_create_chunk_embeddings(documents)


def search_chunked(query: str, limit: int):
    documents = load_movies()
    search_instance = ChunkedSemanticSearch()
    search_instance.load_or_create_chunk_embeddings(documents)
    results = search_instance.search_chunks(query, limit)
    for index, res in enumerate(results):
        print(f"\n{index + 1}. {res['title']} (score: {res['score']})")
        print(f"   {res['document']}...")
