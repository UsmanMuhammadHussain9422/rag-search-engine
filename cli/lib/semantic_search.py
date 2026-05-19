from collections import defaultdict
from operator import itemgetter
import os
from typing import Sequence
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.typing import NDArray
from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        # Store only normalized embeddings for cosine search.
        self.embeddings: NDArray[np.floating] | None = None
        self.documents: Sequence[dict] | None = None
        self.document_map: dict[int, dict] = defaultdict(dict)

        self.movie_embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings_normalized.npy")

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

        raw_embeddings = self.model.encode(movie_strings, show_progress_bar=True, convert_to_numpy=True)
        self.embeddings = self._normalize_embeddings(raw_embeddings)

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

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        if not query or query.isspace():
            raise ValueError("Query must be a non-empty string.")
        if self.documents is None or len(self.documents) == 0:
            raise ValueError("Documents are empty. Call load_or_create_embeddings first.")
        if self.embeddings is None:
            raise ValueError("Normalized embeddings are not loaded. Call load_or_create_embeddings first.")

        # get normalized query vector
        q = self.generate_embedding(query)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            q_norm = 1.0
        q = q / q_norm

        # vectorized dot-product against normalized embeddings -> cosine scores
        scores = np.dot(self.embeddings, q)

        # fast top-k indices (handles case limit >= n)
        n = scores.shape[0]
        k = min(max(1, int(limit)), n)
        if k == n:
            topk_idx = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, k - 1)[:k]
            topk_idx = part[np.argsort(-scores[part])]

        # build results
        results = []
        for idx in topk_idx:
            doc = self.documents[idx]
            results.append({"score": float(scores[idx]), "title": doc["title"], "description": doc["description"]})
        return results
    
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
        print(f"{index + 1}. {res['title']} (score: {res['score']})\n")
