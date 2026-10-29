"""Load and validate named prompt profiles at application startup."""

from dataclasses import dataclass
from pathlib import Path
import re
import tomllib
from typing import Literal


EnhanceMethod = Literal["spell", "rewrite", "expand"]


@dataclass(frozen=True)
class PromptProfile:
    spell: str
    rewrite: str
    expand: str
    rerank: str

    def enhancement_prompt(self, method: EnhanceMethod) -> str:
        if method not in ("spell", "rewrite", "expand"):
            raise ValueError(f"Unsupported enhancement method: {method}")
        return getattr(self, method)


def load_prompt_profile(name: str, directory: Path) -> PromptProfile:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Prompt profile names may contain only letters, numbers, underscores, and hyphens")
    path = directory / f"{name}.toml"
    try:
        with path.open("rb") as file:
            content = tomllib.load(file)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load prompt profile '{name}' from {path}") from exc

    prompts = {}
    for field, section, key in (
        ("spell", "enhancement", "spell"),
        ("rewrite", "enhancement", "rewrite"),
        ("expand", "enhancement", "expand"),
        ("rerank", "reranking", "individual"),
    ):
        group = content.get(section)
        text = group.get(key) if isinstance(group, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Prompt profile '{name}' requires nonblank text at {section}.{key}")
        prompts[field] = text.strip()
    return PromptProfile(**prompts)
