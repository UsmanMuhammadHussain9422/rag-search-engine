"""The LLM contract used by search features, independent of any vendor SDK."""

from typing import Literal, Protocol


class LLMError(RuntimeError):
    """An LLM request failed or returned an unusable response."""


class LLMUnavailableError(RuntimeError):
    """An optional LLM feature was requested without a configured client."""


class LLMClient(Protocol):
    def complete(
        self,
        *,
        system: str,
        user: str,
        output: Literal["text", "json"] = "text",
    ) -> str:
        """Return response text; raise LLMError for transport or response failures."""
        ...


class ManagedLLMClient(LLMClient, Protocol):
    """A client whose lifetime is owned by the application runtime."""

    def close(self) -> None: ...
