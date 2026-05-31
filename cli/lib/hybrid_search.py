import os
import numpy as np

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    

def normalise_scores(scores: list[float]) -> list[float]:
    if not scores: return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0]
    
    return [(normalise_score(score, min_score, max_score)) for score in scores] 

def normalise_score(score, min_score, max_score) -> float:
    return (score - min_score) / (max_score - min_score)

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def normalise_command(scores: list[float]) -> None:
    normalised_scores = normalise_scores(scores)
    for score in normalised_scores:
        print(f"* {score:.4f}")
    