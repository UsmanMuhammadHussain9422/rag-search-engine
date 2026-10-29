from heapq import nlargest
from typing import Any, Sequence

from rag_search.search_models import SearchResult, format_search_result
from rag_search.search.retrievers import Retriever
from rag_search.config.defaults import (
    DEFAULT_ALPHA,
    DEFAULT_K,
    DEFAULT_SEARCH_LIMIT,
    RETRIEVAL_MULTIPLIER,
)


class HybridSearch:
    """Combine initialized retrievers without loading models, data, or indexes."""

    def __init__(self, keyword_search: Retriever, semantic_search: Retriever) -> None:
        self.keyword_search = keyword_search
        self.semantic_search = semantic_search

    def weighted_search(
        self,
        query: str,
        alpha: float = DEFAULT_ALPHA,
        limit: int = DEFAULT_SEARCH_LIMIT,
        candidate_multiplier: int = RETRIEVAL_MULTIPLIER,
    ) -> list[SearchResult]:
        candidate_limit = limit * candidate_multiplier

        bm25_results = self.keyword_search.search(query, candidate_limit)
        semantic_results = self.semantic_search.search(query, candidate_limit)

        bm25_scores = normalise_scores([result["score"] for result in bm25_results])
        semantic_scores = normalise_scores([result["score"] for result in semantic_results])

        results: dict[int | str, dict[str, Any]] = {}

        def add_score(
            search_results: Sequence[SearchResult],
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

    def rrf_search(
        self,
        query: str,
        k: float = DEFAULT_K,
        limit: int = DEFAULT_SEARCH_LIMIT,
        candidate_multiplier: int = RETRIEVAL_MULTIPLIER,
        *,
        candidate_limit: int | None = None,
    ) -> list[SearchResult]:
        if k < 0 or limit < 1 or candidate_multiplier < 1:
            raise ValueError("k must be nonnegative and limits must be positive")
        candidate_limit = max(
            limit,
            candidate_limit if candidate_limit is not None else limit * candidate_multiplier,
        )

        bm25_results = self.keyword_search.search(query, candidate_limit)
        semantic_results = self.semantic_search.search(query, candidate_limit)

        results: dict[int | str, dict[str, Any]] = {}

        def add_score(
            search_results: Sequence[SearchResult], score_key: str, rank_key: str
        ) -> None:
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
                        "semantic_rank": 0,
                    },
                )

                entry[rank_key] = rank
                entry[score_key] = rrf_score(rank, k)

        add_score(bm25_results, "keyword_score", "keyword_rank")
        add_score(semantic_results, "semantic_score", "semantic_rank")

        for result in results.values():
            result["score"] = result["semantic_score"] + result["keyword_score"]

        top_results = nlargest(
            limit,
            results.items(),
            key=lambda item: item[1]["score"],
        )

        return [
            format_search_result(doc_id, **entry)
            for doc_id, entry in top_results
        ]

def normalise_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    return [normalise_score(score, min_score, max_score) for score in scores]

def normalise_score(score: float, min_score: float, max_score: float) -> float:
    return (score - min_score) / (max_score - min_score)

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k: float = DEFAULT_K) -> float:
    return 1 / (k + rank)
