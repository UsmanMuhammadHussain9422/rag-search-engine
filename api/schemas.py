"""Typed HTTP contracts shared by the search API and its Python client."""

from dataclasses import asdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_search.config.defaults import DEFAULT_K, DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT
from rag_search.search_models import SearchResponse


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class SearchRequest(APIModel):
    collection: str = Field(default="movies", min_length=1, max_length=128, strict=True)
    query: str = Field(min_length=1, max_length=10000, strict=True)
    limit: int = Field(default=DEFAULT_SEARCH_LIMIT, ge=1, le=MAX_SEARCH_LIMIT, strict=True)
    k: float = Field(default=DEFAULT_K, ge=0, strict=True)
    enhance: Literal["spell", "rewrite", "expand"] | None = None
    rerank_method: Literal["individual"] | None = None

    @field_validator("query", "collection")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Must not be blank")
        return value


class SearchResultSchema(APIModel):
    id: int | str
    title: str
    document: str
    display_doc: str
    score: float
    metadata: dict[str, Any]
    individual_score: float | None = Field(default=None, ge=0, le=10)


class SearchResponseSchema(APIModel):
    collection: str
    original_query: str
    effective_query: str
    results: list[SearchResultSchema]

    @classmethod
    def from_domain(cls, collection: str, response: SearchResponse) -> "SearchResponseSchema":
        return cls.model_validate({"collection": collection, **asdict(response)})

    def to_domain(self) -> SearchResponse:
        return SearchResponse(
            self.original_query,
            self.effective_query,
            [result.model_dump(exclude_none=True) for result in self.results],
        )


class HealthResponse(APIModel):
    status: Literal["ready"]
    collections: list[str]


class ValidationIssue(APIModel):
    loc: list[str | int]
    msg: str
    type: str


class ErrorResponse(APIModel):
    detail: str | list[ValidationIssue]
