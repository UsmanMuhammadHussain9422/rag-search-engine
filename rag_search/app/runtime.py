"""Own the initialized collections for one application/server lifetime."""

from collections.abc import Callable

from rag_search.search.search_service import SearchService


class CollectionNotFoundError(KeyError):
    pass


class SearchRuntime:
    def __init__(
        self,
        collections: dict[str, SearchService],
        *,
        close: Callable[[], None] | None = None,
    ) -> None:
        self._collections = dict(collections)
        self._close = close

    @property
    def collection_names(self) -> list[str]:
        return sorted(self._collections)

    def get_collection(self, name: str) -> SearchService:
        try:
            return self._collections[name]
        except KeyError:
            raise CollectionNotFoundError(name) from None

    def close(self) -> None:
        if self._close is not None:
            close, self._close = self._close, None
            close()
