"""Manual, opt-in LLM smoke check. Importing this module makes no API calls."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main() -> None:
    from rag_search.llm.client_factory import create_llm_client
    from rag_search.config.settings import Settings

    settings = Settings.from_environment()
    client = create_llm_client(settings)
    if client is None:
        raise RuntimeError("Set OPENROUTER_API_KEY or LLM_API_KEY first")
    try:
        print(client.complete(
            system="Answer in one short paragraph.",
            user="What is reciprocal rank fusion?",
        ))
    finally:
        client.close()


if __name__ == "__main__":
    main()
