"""A common retriever contract and adapters for the initialized search indexes."""

from threading import Lock
from typing import TYPE_CHECKING, Protocol

from rag_search.search_models import SearchResult

if TYPE_CHECKING:
    from rag_search.search.keyword_search import InvertedIndex
    from rag_search.search.semantic_search import ChunkedSemanticSearch


class Retriever(Protocol):
    def search(self, query: str, limit: int) -> list[SearchResult]: ...


class BM25Retriever:
    def __init__(self, index: "InvertedIndex") -> None:
        self.index = index

    def search(self, query: str, limit: int) -> list[SearchResult]:
        return self.index.bm25_search(query, limit)


class SemanticRetriever:
    def __init__(self, index: "ChunkedSemanticSearch") -> None:
        self.index = index
        self._lock = Lock()

    def search(self, query: str, limit: int) -> list[SearchResult]:
        # Serialize access to the shared embedding model across API threads.
        with self._lock:
            return self.index.search_chunks(query, limit)
