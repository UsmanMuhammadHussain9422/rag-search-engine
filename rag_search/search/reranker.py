"""Individual relevance scoring, independent of the LLM provider and transport."""

import json
import math
from typing import Protocol, Sequence

from rag_search.llm.llm_client import LLMClient, LLMError
from rag_search.config.prompt_profiles import PromptProfile

from rag_search.search_models import SearchResult
from rag_search.config.defaults import DEFAULT_SEARCH_LIMIT


class Reranker(Protocol):
    def rerank(
        self, query: str, documents: Sequence[SearchResult], limit: int
    ) -> list[SearchResult]: ...


class IndividualLLMReranker:
    def __init__(self, llm: LLMClient, *, prompts: PromptProfile) -> None:
        self.llm = llm
        self.prompts = prompts

    def rate_document(self, query: str, title: str, text: str) -> float:
        content = self.llm.complete(
            system=self.prompts.rerank,
            user=json.dumps({"query": query, "title": title, "text": text}),
            output="json",
        )
        try:
            result = json.loads(content)
            score = result["score"]
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("Score must be a number")
            if not math.isfinite(score) or not 0 <= score <= 10:
                raise ValueError("Score must be between 0 and 10")
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            raise LLMError("The LLM returned an invalid relevance score.") from exc
        return float(score)

    def rerank(
        self,
        query: str,
        documents: Sequence[SearchResult],
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[SearchResult]:
        scored: list[SearchResult] = [
            {**doc, "individual_score": self.rate_document(query, doc["title"], doc["document"])}
            for doc in documents
        ]
        scored.sort(key=lambda doc: (doc["individual_score"], doc["score"]), reverse=True)
        return scored[:limit]
