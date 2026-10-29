"""OpenAI-compatible transport. SDK types do not escape this adapter."""

from typing import Literal

from openai import OpenAI, OpenAIError

from rag_search.llm.llm_client import LLMError


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        max_tokens: int = 1024,
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.client = client if client is not None else OpenAI(
            api_key=api_key, base_url=base_url, timeout=30.0, max_retries=2
        )

    def complete(
        self,
        *,
        system: str,
        user: str,
        output: Literal["text", "json"] = "text",
    ) -> str:
        if output not in ("text", "json"):
            raise ValueError(f"Unsupported LLM output format: {output}")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=self.max_tokens,
                response_format={"type": "json_object" if output == "json" else "text"},
            )
        except OpenAIError as exc:
            raise LLMError("The LLM request failed.") from exc

        if not response.choices:
            raise LLMError("The LLM returned no choices.")
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise LLMError("The LLM did not return a complete text response.")
        content = choice.message.content
        if not content or not content.strip():
            raise LLMError("The LLM returned empty text.")
        return content.strip()

    def close(self) -> None:
        self.client.close()
