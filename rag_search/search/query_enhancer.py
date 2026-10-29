"""Stateless query enhancement using an injected LLM client."""

from rag_search.llm.llm_client import LLMClient
from rag_search.config.prompt_profiles import EnhanceMethod, PromptProfile


class QueryEnhancer:
    def __init__(self, llm: LLMClient, *, prompts: PromptProfile) -> None:
        self.llm = llm
        self.prompts = prompts

    def enhance_query(self, query: str, method: EnhanceMethod | None = None) -> str:
        if method is None:
            return query
        text = self.llm.complete(
            system=self.prompts.enhancement_prompt(method),
            user=query,
            output="text",
        ).strip().strip('"').strip()
        if not text:
            return query
        return f"{query} {text}" if method == "expand" else text

    def spell_correct(self, query: str) -> str:
        return self.enhance_query(query, "spell")

    def rewrite_query(self, query: str) -> str:
        return self.enhance_query(query, "rewrite")

    def expand_query(self, query: str) -> str:
        return self.enhance_query(query, "expand")
