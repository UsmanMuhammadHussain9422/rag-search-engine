"""A typed search HTTP client with connection pooling and explicit ownership."""

import httpx

from api.schemas import SearchRequest, SearchResponseSchema
from rag_search.search_models import SearchResponse


class SearchHTTPClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(
            base_url=base_url.rstrip("/") + "/", timeout=timeout,
        )

    def search(self, query: str, *, collection: str = "movies", **options) -> SearchResponse:
        payload = SearchRequest(query=query, collection=collection, **options)
        try:
            response = self._client.post("search", json=payload.model_dump(mode="json", exclude_none=True))
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", exc.response.reason_phrase)
            except (ValueError, AttributeError):
                detail = exc.response.reason_phrase
            raise RuntimeError(f"Search server returned HTTP {exc.response.status_code}: {detail}") from None
        except httpx.RequestError as exc:
            raise RuntimeError(f"Could not reach the search server: {exc}") from None
        try:
            result = SearchResponseSchema.model_validate(response.json())
        except ValueError as exc:
            raise RuntimeError("Search server returned an invalid response") from exc
        return result.to_domain()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "SearchHTTPClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
