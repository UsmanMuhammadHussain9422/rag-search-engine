"""Application workflow. No CLI output, model loading, or vendor SDK imports."""

import math
import logging
from typing import Literal

from rag_search.llm.llm_client import LLMUnavailableError

from rag_search.search_models import SearchResponse
from rag_search.search.hybrid_search import HybridSearch
from rag_search.config.prompt_profiles import EnhanceMethod
from rag_search.search.query_enhancer import QueryEnhancer
from rag_search.search.reranker import Reranker
from rag_search.config.defaults import (
    DEFAULT_K, DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT,
    RETRIEVAL_MULTIPLIER, SEARCH_MULTIPLIER,
)


logger = logging.getLogger(__name__)


class SearchService:
    def __init__(
        self,
        search: HybridSearch,
        *,
        enhancer: QueryEnhancer | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.search_engine = search
        self.enhancer = enhancer
        self.reranker = reranker

    def search(
        self,
        query: str,
        *,
        k: float = DEFAULT_K,
        limit: int = DEFAULT_SEARCH_LIMIT,
        enhance: EnhanceMethod | None = None,
        rerank_method: Literal["individual"] | None = None,
    ) -> SearchResponse:
        if not query.strip():
            raise ValueError("Query must be a non-empty string")
        if not 1 <= limit <= MAX_SEARCH_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_SEARCH_LIMIT}")
        if not math.isfinite(k) or k < 0:
            raise ValueError("k must be finite and nonnegative")
        if enhance not in (None, "spell", "rewrite", "expand"):
            raise ValueError(f"Unsupported enhancement method: {enhance}")
        if rerank_method not in (None, "individual"):
            raise ValueError(f"Unsupported rerank method: {rerank_method}")
        if enhance and self.enhancer is None:
            raise LLMUnavailableError("Query enhancement requires a configured LLM client.")
        if rerank_method and self.reranker is None:
            raise LLMUnavailableError("LLM reranking requires a configured LLM client.")

        effective_query = query
        if enhance and self.enhancer is not None:
            effective_query = self.enhancer.enhance_query(query, enhance)

        pool_limit = limit * SEARCH_MULTIPLIER if rerank_method else limit
        # Retrieval depth and reranking depth are independent, not compounded.
        candidate_limit = max(pool_limit, limit * RETRIEVAL_MULTIPLIER)
        logger.debug(
            "Searching for %r with k=%s, limit=%s, pool_limit=%s, candidate_limit=%s",
            effective_query, k, limit, pool_limit, candidate_limit,
        )
        results = self.search_engine.rrf_search(
            effective_query, k, pool_limit, candidate_limit=candidate_limit
        )
        if rerank_method and self.reranker is not None:
            results = self.reranker.rerank(effective_query, results, limit)
        return SearchResponse(query, effective_query, results[:limit])
