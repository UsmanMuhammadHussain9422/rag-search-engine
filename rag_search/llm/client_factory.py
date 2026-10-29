"""Select the LLM adapter in one place, without coupling search features to an SDK."""

from collections.abc import Callable

from rag_search.config.settings import Settings
from rag_search.llm.llm_client import ManagedLLMClient


def _openai_compatible_client(settings: Settings) -> ManagedLLMClient | None:
    if not settings.llm_api_key:
        return None
    from rag_search.llm.openai_compatible_client import OpenAICompatibleClient
    return OpenAICompatibleClient(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        max_tokens=settings.llm_max_tokens,
    )


CLIENT_FACTORIES: dict[str, Callable[[Settings], ManagedLLMClient | None]] = {
    "openai_compatible": _openai_compatible_client,
}


def create_llm_client(settings: Settings) -> ManagedLLMClient | None:
    try:
        factory = CLIENT_FACTORIES[settings.llm_backend]
    except KeyError:
        raise ValueError(f"Unsupported LLM_BACKEND: {settings.llm_backend}") from None
    return factory(settings)
