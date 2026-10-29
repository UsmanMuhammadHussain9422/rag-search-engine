"""Read configuration once at the application boundary."""

from dataclasses import dataclass, field
import os
from pathlib import Path
import tomllib

from dotenv import load_dotenv

from rag_search.config.defaults import CACHE_DIR, PROJECT_ROOT

CONFIG_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CollectionSettings:
    prompt_profile: str


@dataclass(frozen=True)
class Settings:
    embedding_model: str = "all-MiniLM-L6-v2"
    movie_cache_dir: Path = Path(CACHE_DIR) / "collections" / "movies"
    llm_model: str = "openrouter/free"
    llm_backend: str = "openai_compatible"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str | None = field(default=None, repr=False)
    llm_max_tokens: int = 1024
    collections_file: Path = CONFIG_DIR / "collections.toml"
    prompts_dir: Path = CONFIG_DIR / "prompts"

    def collection(self, name: str) -> CollectionSettings:
        """Read collection policy once during startup, never during a query."""
        try:
            with self.collections_file.open("rb") as file:
                content = tomllib.load(file)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not load collection settings from {self.collections_file}") from exc
        collections = content.get("collections")
        entry = collections.get(name) if isinstance(collections, dict) else None
        profile = entry.get("prompt_profile") if isinstance(entry, dict) else None
        if not isinstance(profile, str) or not profile.strip():
            raise ValueError(f"Collection '{name}' requires a prompt_profile setting")
        return CollectionSettings(prompt_profile=profile)

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv(Path(PROJECT_ROOT) / ".env")
        max_tokens = int(os.environ.get("LLM_MAX_TOKENS") or "1024")
        if max_tokens < 1:
            raise ValueError("LLM_MAX_TOKENS must be positive")
        return cls(
            embedding_model=os.environ.get("EMBEDDING_MODEL") or "all-MiniLM-L6-v2",
            llm_model=os.environ.get("LLM_MODEL") or "openrouter/free",
            llm_backend=os.environ.get("LLM_BACKEND") or "openai_compatible",
            llm_base_url=os.environ.get("LLM_BASE_URL") or "https://openrouter.ai/api/v1",
            llm_api_key=os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY"),
            llm_max_tokens=max_tokens,
            collections_file=Path(os.environ.get("SEARCH_COLLECTIONS_CONFIG") or cls.collections_file),
            prompts_dir=Path(os.environ.get("SEARCH_PROMPTS_DIR") or cls.prompts_dir),
        )
