"""Embedding indexes, passage chunking, and semantic retrieval."""

from collections import defaultdict
import json
import os
import re
from typing import Sequence, cast

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from rag_search.config.defaults import (
    CACHE_DIR, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT, DEFAULT_SEMANTIC_CHUNK_SIZE,
)
from rag_search.search_models import ChunkMetadata, SearchResult, format_search_result


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2", *, model=None, cache_dir: str = CACHE_DIR) -> None:
        self.model = model if model is not None else SentenceTransformer(model_name)
        self.movie_embeddings_path = os.path.join(cache_dir, "movie_embeddings.npy")
        self.chunk_embeddings_path = os.path.join(cache_dir, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(cache_dir, "chunk_metadata.json")
        os.makedirs(cache_dir, exist_ok=True)
        # Store only normalized embeddings for cosine search.
        self.embeddings: NDArray[np.floating] | None = None
        self.documents: Sequence[dict] | None = None
        self.document_map: dict[int, dict] = defaultdict(dict)

    def _normalize_embeddings(self, embeddings: NDArray[np.floating]) -> NDArray[np.floating]:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def generate_embedding(self, text: str) -> NDArray[np.floating]:
        if not text or text.isspace():
            raise ValueError("Cannot generate embeddings for empty text.")

        return self.model.encode([text], convert_to_numpy=True)[0]

    def build_embeddings(self, documents: Sequence[dict]) -> NDArray[np.floating]:
        self.documents = documents
        self.document_map = defaultdict(dict)
        movie_strings = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True, convert_to_numpy=True)

        with open(self.movie_embeddings_path, 'wb') as file:
            np.save(file, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: Sequence[dict]) -> NDArray[np.floating]:
        self.documents = documents
        self.document_map = defaultdict(dict)
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.movie_embeddings_path):
            self.embeddings = np.load(self.movie_embeddings_path, allow_pickle=False)
            if (self.embeddings is not None and self.documents is not None and
                len(self.embeddings) == len(self.documents)):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[SearchResult]:
        if not query or query.isspace():
            raise ValueError("Query must be a non-empty string.")
        if self.documents is None or len(self.documents) == 0:
            raise ValueError("Documents are empty. Call load_or_create_embeddings first.")
        if self.embeddings is None:
            raise ValueError("Normalized embeddings are not loaded. Call load_or_create_embeddings first.")

        query_embedding = self.generate_embedding(query)

        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, self.documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in similarities[:limit]:
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"],
                    score=score,
                )
            )

        return results


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", *, model=None, cache_dir: str = CACHE_DIR) -> None:
        super().__init__(model_name, model=model, cache_dir=cache_dir)
        self.chunk_embeddings = None
        self.chunk_metadata: list[ChunkMetadata] | None = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        if documents is None:
            raise ValueError("Cannot build chunk embeddings for empty Document.")

        self.documents = documents
        self.document_map = defaultdict(dict)

        all_chunks = []
        self.chunk_metadata = []

        for doc_idx, doc in enumerate(self.documents):
            if not str(doc["description"]).strip():
                continue
            chunks = get_semantic_text_chunks(doc["description"], chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                self.chunk_metadata.append({"movie_idx": doc_idx, "chunk_idx": chunk_idx, "total_chunks": len(chunks)})

        if not all_chunks:
            raise ValueError("No non-empty text is available to embed.")
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)

        with open(self.chunk_embeddings_path, 'wb') as file:
            np.save(file, self.chunk_embeddings)

        with open(self.chunk_metadata_path, 'w') as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = defaultdict(dict)
        total_chunks: int = 0
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.chunk_metadata_path):
            with open(self.chunk_metadata_path, "r") as file:
                metadata = json.load(file)
                self.chunk_metadata = cast(list[ChunkMetadata], metadata["chunks"])
                total_chunks = metadata["total_chunks"]

        if os.path.exists(self.chunk_embeddings_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path, allow_pickle=False)
            if (self.chunk_embeddings is not None and self.documents is not None and self.chunk_metadata is not None and
                len(self.chunk_embeddings) == total_chunks):
                return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[SearchResult]:
        if not query or query.isspace():
            raise ValueError("Query must be a non-empty string.")
        if self.documents is None or len(self.documents) == 0:
            raise ValueError("Documents are empty. Call load_or_create_embeddings first.")
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("No chunk embeddings loaded. Call load_or_create_chunk_embeddings first.")

        query_embedding = self.generate_embedding(query)

        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "score": similarity,
                    "metadata": self.chunk_metadata[i],
                }
            )

        movie_scores = defaultdict(int)
        movie_metadata:dict[int, ChunkMetadata] = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if (
                movie_idx not in movie_scores
                or chunk_score["score"] > movie_scores[movie_idx]
            ):
                movie_scores[movie_idx] = chunk_score["score"]
                movie_metadata[movie_idx] = chunk_score["metadata"]

        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for movie_idx, score in sorted_movies[:limit]:
            if movie_idx is None:
                continue
            doc = self.documents[movie_idx]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"],
                    score=score,
                    **movie_metadata[movie_idx],
                )
            )

        return results


def get_text_chunks(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")
    words = text.split()
    if not words:
        return []

    return create_text_chunks(chunk_size, overlap, words)


def create_text_chunks(chunk_size: int, overlap: int, text_strings: list[str]):
    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(text_strings), step):
        if (len(text_strings) != 1) and (start == (len(text_strings) - 1)): break
        chunks.append(" ".join(text_strings[start:start + chunk_size]))

    return chunks


def get_semantic_text_chunks(text: str, chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    if chunk_size < 1:
        raise ValueError("max_chunk_size must be at least 1.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than max_chunk_size.")

    sentence_regex = r"(?<=[.!?])\s+"
    clean_text = text.strip()
    if clean_text.isspace(): return []
    sentences = re.split(sentence_regex, clean_text)

    if not sentences: return []

    if len(sentences) == 1 and sentences[0] and sentences[0][-1] not in ".!?":
        return create_text_chunks(chunk_size, overlap, [clean_text])

    cleaned_sentences = []
    for sentence in sentences:
        clean_sentence = sentence.strip()
        if not clean_sentence.isspace():
            cleaned_sentences.append(clean_sentence)

    return create_text_chunks(chunk_size, overlap, sentences)
