import os
import json

from heapq import nlargest
from typing import Any, Literal, TypedDict

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import (
    BLUE,
    GREEN,
    RESET,
    DEFAULT_ALPHA,
    DEFAULT_K,
    DEFAULT_SEARCH_LIMIT,
    SEARCH_MULTIPLIER,
    SearchResult,
    format_search_result,
    load_movies,
    set_color,
)
from .query_enhancer import Query_Enhancer
from .reranker import RerankedSearchResult, rerank

from llm.openrouter import openrouter
from openai.types.chat import ChatCompletionMessageParam

DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"

class RRFSearchCommandResult(TypedDict):
    original_query: str
    enhanced_query: str | None
    enhance_method: Literal["spell", "expand", "rewrite"] | None
    query: str
    k: int
    rerank_method: Literal["individual", "batch", "cross_encoder"] | None
    reranked: bool
    results: list[SearchResult] | list[RerankedSearchResult]

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

def build_command() -> None:
    documents = load_movies()
    hse = HybridSearch(documents)

def normalise_command(scores: list[float]) -> None:
    normalised_scores = normalise_scores(scores)
    for score in normalised_scores:
        print(f"* {score:.4f}")

def weighted_search_command(query: str, k: int, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    documents = load_movies()
    hse = HybridSearch(documents)
    results = hse.weighted_search(query, k, limit)
    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(
            f"{index}. {result['title']}\n"
            f"  Hybrid Score: {result['score']:.3f}\n"
            f"  BM25: {metadata['keyword_score']:.3f} "
            f"Semantic: {metadata['semantic_score']:.3f}\n"
            f"  {result['document']}\n"
        )


def _print_rrf_results(
    results: list[SearchResult] | list[RerankedSearchResult],
    *,
    heading: str,
    color: str | None = None,
) -> None:
    try:
        if color:
            set_color(color)
        print(f"{heading}\n")
        if not results:
            print("No results found.\n")
        for index, result in enumerate(results, start=1):
            metadata = result["metadata"]
            print(f"{index}. {result['title']}")
            if "individual_score" in result:
                print(f"   Re-rank Score: {result['individual_score']:.1f}")
            if "batch_rank" in result:
                print(f"   Re-rank Rank: {result['batch_rank']}")
            if "cross_encoding_rank" in result:
                print(f"   Cross-encoder Score: {result['cross_encoding_rank']:.3f}")
            print(f"   RRF Score: {result['score']:.6f}")
            print(
                f"   BM25 Rank: {metadata.get('keyword_rank', 0)}, "
                f"Semantic Rank: {metadata.get('semantic_rank', 0)}"
            )
            print(f"   {result['preview']}\n")
    finally:
        if color:
            set_color(RESET)

def format_rrf_results(results: list[SearchResult]) -> list[str]:
    """Return one string per result with its full document, score, and ranks."""
    formatted_results: list[str] = []
    for index, result in enumerate(results, 1):
        formatted_results.append(
            f"{index}. Title: {result['title']}\n"
            f"Document: {result['document']}\n"
        )
    return formatted_results

def llm_evaluate_results(query: str, formatted_results: list)  -> list:
    OpenRouter = openrouter()
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {chr(10).join(formatted_results)}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers other than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

    [2, 0, 3, 2, 0, 1]"""

    messages: ChatCompletionMessageParam = {
        "role": "user",
        "content": ""
    }
    messages["content"] = prompt
    response = OpenRouter.trigger_prompt(prompts=[messages])
    parsed_scores = json.loads((response.choices[0].message.content or "").strip())
    if len(parsed_scores) == len(formatted_results):
        return parsed_scores

    raise ValueError(
        f"LLM response parsing error. Expected {len(formatted_results)} scores, got {len(parsed_scores)}. Response: {parsed_scores}"
    )

def rrf_search_command(
    original_query: str,
    *,
    k: int = DEFAULT_K,
    limit: int = DEFAULT_SEARCH_LIMIT,
    enhance_type: Literal["spell", "rewrite", "expand"] | None = None,
    rerank_method: Literal["individual", "batch", "cross_encoder"] | None = None,
    evaluate: bool = False,
    verbose: bool = True,
) -> RRFSearchCommandResult:
    # Enhance the query when an enhancement method is selected.
    query = original_query
    enhanced_query = query
    if enhance_type:
        query_enhancer = Query_Enhancer()
        enhanced_query = query_enhancer.enhance_query(query, enhance_type)
        query = enhanced_query
        if verbose:
            print(
                f"Enhanced query ({enhance_type}): "
                f"'{original_query}' -> '{enhanced_query}'\n"
            )

    # Retrieve extra RRF candidates when reranking is requested.
    documents = load_movies()
    hse = HybridSearch(documents)
    search_limit = limit * SEARCH_MULTIPLIER if rerank_method else limit
    results = hse.rrf_search(query, k, search_limit)

    # Show the candidates before reranking when debug output is enabled.
    if verbose and DEBUG_MODE and rerank_method:
        _print_rrf_results(
            results,
            heading=f"[Debug] RRF candidates before reranking ({len(results)} results):",
            color=GREEN,
        )

    # Rerank the candidates and keep the requested number of results.
    reranked = bool(rerank_method)
    if rerank_method:
        if verbose:
            print(f"Re-ranking {len(results)} candidates using {rerank_method}...\n")
        results = rerank(query, results, method=rerank_method, limit=limit)

    # Display the final results with consistent scores, ranks, and spacing.
    if verbose:
        _print_rrf_results(
            results,
            heading=f"Reciprocal Rank Fusion Results for '{query}' (k={k}):",
            color=BLUE if DEBUG_MODE else None,
        )

    if evaluate:
        formatted_results = format_rrf_results(results)
        rated_scores = llm_evaluate_results(original_query, formatted_results)
        print(f"- Query: {original_query}")
        for index, res in enumerate(results):
            print(f"{index + 1}. {res['title']}: {rated_scores[index]}/3")

    # Return the query details and results for programmatic callers.
    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhance_method": enhance_type,
        "query": query,
        "k": k,
        "rerank_method": rerank_method,
        "reranked": reranked,
        "results": results,
    }
