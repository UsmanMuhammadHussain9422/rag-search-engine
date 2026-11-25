from typing import Literal

from lib.reranker import RerankedSearchResult
from lib.search_utils import DEFAULT_K, DEFAULT_SEARCH_LIMIT, RESET, SearchResult, set_color
from .hybrid_search import rrf_search_command

from llm.openrouter import openrouter
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam

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

def augmented_search_command(
    query: str,
    *,
    k: int = DEFAULT_K,
    limit: int = DEFAULT_SEARCH_LIMIT,
    enhance_type: Literal["spell", "rewrite", "expand"] | None = None,
    rerank_method: Literal["individual", "batch", "cross_encoder"] | None = None,
    verbose: bool = False
) -> str:

    rrf_search_result = rrf_search_command(query, k=k, limit=limit, enhance_type=enhance_type, rerank_method=rerank_method, verbose=verbose)

    results = rrf_search_result["results"]
    doc_map: dict[int, SearchResult] = {}
    docs = ""
    rrf_search_results_text = "Search Results:\n"
    for result in results:
        doc_id = result["id"]
        doc_map[doc_id] = result
        docs += f"{doc_id}: {result.get('title', '')} - {result.get('document', '')[:200]}\n"
        rrf_search_results_text += f"- {result.get('title', '')}\n"

    rrf_search_results_text.removesuffix('\n')

    if not rrf_search_results_text or rrf_search_results_text.isspace():
        return "No results found"

    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {docs}

    Answer:"""

    rag_response = "RAG Response:\n"

    llm = openrouter()
    prompts:list[ChatCompletionMessageParam] = []
    prompts.append(
        ChatCompletionUserMessageParam(content= prompt, role="user")
    )
    response = llm.trigger_prompt(prompts=prompts)
    if response and response.choices and response.choices[0] and response.choices[0].message and response.choices[0].message.content:
        rag_response += response.choices[0].message.content

    return f"{rrf_search_results_text}\n{rag_response}"