"""Read the current movie JSON source."""

import json

from rag_search.config.defaults import DATA_PATH


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]
