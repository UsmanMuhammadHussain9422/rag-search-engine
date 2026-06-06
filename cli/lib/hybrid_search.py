from heapq import nlargest
import math
import os
from typing import Any
from .custom_types import SearchResult

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import DEFAULT_SEARCH_LIMIT, DEFAULT_ALPHA, DEFAULT_K, load_movies, format_search_result

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

    def weighted_search(self, query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT, candidate_multiplier: int = 500) -> list[SearchResult]:

        candidate_limit = limit * candidate_multiplier

        bm25_results = self._bm25_search(query, candidate_limit)
        semantic_results = self.semantic_search.search_chunks(query, candidate_limit)

        bm25_scores = normalise_scores([result["score"] for result in bm25_results])
        semantic_scores = normalise_scores([result["score"] for result in semantic_results])

        results: dict[int, dict[str, Any]] = {}

        def add_score(
            search_results: list[dict[str, Any]],
            scores: list[float],
            score_key: str,
        ) -> None:
            for result, score in zip(search_results, scores):
                doc_id = result["id"]

                entry = results.setdefault(
                    doc_id,
                    {
                        "title": result["title"],
                        "document": result["document"],
                        "keyword_score": 0.0,
                        "semantic_score": 0.0,
                    },
                )

                entry[score_key] = score

        add_score(bm25_results, bm25_scores, "keyword_score")
        add_score(semantic_results, semantic_scores, "semantic_score")

        for entry in results.values():
            entry["score"] = hybrid_score(
                entry["keyword_score"],
                entry["semantic_score"],
                alpha,
            )

        top_results = nlargest(
            limit,
            results.items(),
            key=lambda item: item[1]["score"],
        )
        
        return [
            format_search_result(doc_id, **entry)
            for doc_id, entry in top_results
        ]

    def rrf_search(self, query: str, k: int = DEFAULT_K, limit: int = DEFAULT_SEARCH_LIMIT, candidate_multiplier: int = 500) -> list[SearchResult]:
        candidate_limit = limit * candidate_multiplier

        bm25_results = self._bm25_search(query, candidate_limit)
        semantic_results = self.semantic_search.search_chunks(query, candidate_limit)

        results: dict[int, dict[str, Any]] = {}

        def add_score(search_results: list[dict[str, Any]], score_key, rank_key: str) -> None:
            for index, result in enumerate(search_results, start=1):
                doc_id = result["id"]
                rank = index
                entry = results.setdefault(
                    doc_id,
                    {
                        "title": result["title"],
                        "document": result["document"],
                        "keyword_score": 0.0,
                        "keyword_rank": 0,
                        "semantic_score": 0.0,
                        "semantic_rank": 0
                    },
                )

                entry[rank_key] = rank
                entry[score_key] = rrf_score(rank, k)

        add_score(bm25_results, "keyword_score", "keyword_rank")
        add_score(semantic_results, "semantic_score", "semantic_rank")

        for result in results.values():
            result['score'] = result["semantic_score"] + result["keyword_score"]

        top_results = nlargest(
            limit,
            results.items(),
            lambda item: item[1]["score"]
        )

        return [
            format_search_result(doc_id, **entry)
            for doc_id, entry in top_results
        ]

def normalise_scores(scores: list[float]) -> list[float]:
    if not scores: return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)
    
    return [(normalise_score(score, min_score, max_score)) for score in scores] 

def normalise_score(score, min_score, max_score) -> float:
    return (score - min_score) / (max_score - min_score)

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

def normalise_command(scores: list[float]) -> None:
    normalised_scores = normalise_scores(scores)
    for score in normalised_scores:
        print(f"* {score:.4f}")

def weighted_search_command(query: str, k: int, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    documents = load_movies()
    hse = HybridSearch(documents)
    results = hse.weighted_search(query, k, limit)
    for index, result in enumerate(results):
        print(f"{index + 1 }. {result['title']}\n\tHybrid Score: {result['score']:.3f}\n\tBM25: {result['metadata']['keyword_score']:.3f} Semantic: {result['metadata']['semantic_score']:.3f}\n\t{result['document']}\n".expandtabs(2))

def rrf_search_command(query: str, k: int = DEFAULT_K, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    documents = load_movies()
    hse = HybridSearch(documents)
    results = hse.rrf_search(query, k, limit)
    for index, result in enumerate(results):
        print(f"{index + 1 }. {result['title']}\n\tRRF Score: {result['score']:.3f}\n\tBM25 Rank: {result['metadata']['keyword_rank']:.3f}, Semantic Rank: {result['metadata']['semantic_rank']:.3f}\n\t{result['document']}\n".expandtabs(2))