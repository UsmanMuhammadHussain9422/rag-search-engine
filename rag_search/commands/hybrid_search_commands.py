"""Hybrid command implementations; argument parsing stays in cli/."""

from rag_search.search_models import SearchResponse, SearchResult
from rag_search.search.hybrid_search import normalise_scores
from rag_search.config.prompt_profiles import EnhanceMethod
from rag_search.config.defaults import DEFAULT_K, DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT


def print_search_response(response: SearchResponse, enhance: str | None = None) -> None:
    if enhance:
        print(f"Enhanced query ({enhance}): '{response.original_query}' -> '{response.effective_query}'\n")
    for index, result in enumerate(response.results, start=1):
        print(f"{index}. {result['title']}")
        if "individual_score" in result:
            print(f"  Re-rank Score: {result['individual_score']:.1f}")
        metadata = result["metadata"]
        print(f"  RRF Score: {result['score']:.6f}")
        print(f"  BM25 Rank: {metadata.get('keyword_rank', 0)}, Semantic Rank: {metadata.get('semantic_rank', 0)}")
        print(f"  {result['display_doc']}\n")


def rrf_search_command(
    original_query: str,
    k: float = DEFAULT_K,
    limit: int = DEFAULT_SEARCH_LIMIT,
    enhance_type: EnhanceMethod | None = None,
    rerank_method=None,
) -> list[SearchResult]:
    from rag_search.app.bootstrap import build_runtime

    runtime = build_runtime()
    try:
        response = runtime.get_collection("movies").search(
            original_query, k=k, limit=limit, enhance=enhance_type, rerank_method=rerank_method
        )
        print_search_response(response, enhance_type)
        return response.results
    finally:
        runtime.close()


def build_command() -> None:
    from rag_search.app.bootstrap import build_movie_search
    from rag_search.config.settings import Settings

    build_movie_search(Settings.from_environment())


def normalise_command(scores: list[float]) -> None:
    for score in normalise_scores(scores):
        print(f"* {score:.4f}")


def weighted_search_command(query: str, alpha: float, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    from rag_search.app.bootstrap import build_movie_search
    from rag_search.config.settings import Settings

    if not query.strip() or not 0 <= alpha <= 1 or not 1 <= limit <= MAX_SEARCH_LIMIT:
        raise ValueError("Provide a nonblank query, alpha between 0 and 1, and limit between 1 and 100")
    search = build_movie_search(Settings.from_environment())
    for index, result in enumerate(search.weighted_search(query, alpha, limit), start=1):
        print(f"{index}. {result['title']}")
        print(f"  Hybrid Score: {result['score']:.3f}")
        print(f"  BM25: {result['metadata']['keyword_score']:.3f} Semantic: {result['metadata']['semantic_score']:.3f}")
        print(f"  {result['display_doc']}\n")
