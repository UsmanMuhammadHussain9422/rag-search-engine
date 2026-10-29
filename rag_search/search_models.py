"""Shared result contracts for the search core, HTTP API, and CLI."""

from dataclasses import dataclass
from typing import TypedDict, Any, NotRequired

from rag_search.config.defaults import DOCUMENT_PREVIEW_LENGTH

class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int

class SearchResult(TypedDict):
    id: int | str
    title: str
    document: str
    display_doc: str
    score: float
    metadata: dict[str, Any]
    individual_score: NotRequired[float]


@dataclass
class SearchResponse:
    original_query: str
    effective_query: str
    results: list[SearchResult]


def format_search_result(doc_id: int | str, title: str, document: str, score: float, **metadata: Any) -> SearchResult:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "display_doc": document[:DOCUMENT_PREVIEW_LENGTH],
        "score": float(score),
        "metadata": metadata or {}
    }
