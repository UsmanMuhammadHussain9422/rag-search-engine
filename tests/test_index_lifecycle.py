from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, mock_open, patch

import numpy as np

from rag_search.app.bootstrap import build_runtime
from rag_search.search.keyword_search import InvertedIndex
from rag_search.search.tokenization import load_stopwords
from rag_search.config.settings import Settings


class FakeEmbeddingModel:
    """Deterministic local vectors; tests never download a model."""

    def __init__(self):
        self.batches = []

    def encode(self, texts, **kwargs):
        self.batches.append(list(texts))
        return np.asarray([
            [text.lower().count(term) + 0.01 for term in ("bear", "space", "ocean")]
            for text in texts
        ], dtype=np.float32)


class IndexLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = Settings(movie_cache_dir=Path(self.directory.name))
        self.documents = [
            {"id": 1, "title": "Forest", "description": "A bear lives in the forest. " * 12},
            {"id": 2, "title": "Voyage", "description": "A space journey past distant planets."},
        ]
        self.model = FakeEmbeddingModel()
        self.movies = self.enterContext(patch("rag_search.app.bootstrap.load_movies", return_value=self.documents))
        self.enterContext(patch("rag_search.app.bootstrap.load_stopwords", return_value=[]))
        self.enterContext(patch("rag_search.search.tokenization.load_stopwords", return_value=[]))

    def test_repeated_queries_do_not_reload_indexes_or_source_data(self):
        runtime = build_runtime(self.settings, model=self.model)
        self.addCleanup(runtime.close)
        service = runtime.get_collection("movies")
        index = service.search_engine.keyword_search.index
        index.load = Mock(side_effect=AssertionError("Unexpected index reload"))
        self.movies.side_effect = AssertionError("Unexpected source reload")
        startup_batches = len(self.model.batches)
        for query in ("bear", "space"):
            self.assertTrue(service.search(query).results)
        self.assertEqual(len(self.model.batches), startup_batches + 2)
        index.load.assert_not_called()

    def test_cache_reuse_and_same_size_content_change(self):
        first = build_runtime(self.settings, model=self.model)
        first.close()
        self.assertEqual(len(self.model.batches), 1)
        second = build_runtime(self.settings, model=self.model)
        second.close()
        self.assertEqual(len(self.model.batches), 1, "Unchanged documents should reuse embeddings")
        self.documents[0]["description"] = "An ocean expedition searches for a lost ship."
        third = build_runtime(self.settings, model=self.model)
        self.addCleanup(third.close)
        self.assertEqual(len(self.model.batches), 2)
        self.assertEqual(third.get_collection("movies").search("ocean").results[0]["id"], 1)

    def test_model_change_invalidates_cache(self):
        build_runtime(self.settings, model=self.model).close()
        changed = Settings(movie_cache_dir=self.settings.movie_cache_dir, embedding_model="another-model")
        build_runtime(changed, model=self.model).close()
        self.assertEqual(len(self.model.batches), 2)

    def test_semantic_only_candidates_keep_full_text(self):
        runtime = build_runtime(self.settings, model=self.model)
        self.addCleanup(runtime.close)
        service = runtime.get_collection("movies")
        # No keyword matches, so the text must come from semantic retrieval alone.
        results = service.search("unmatchedword").results
        forest = next(result for result in results if result["id"] == 1)
        self.assertEqual(forest["document"], self.documents[0]["description"])
        self.assertEqual(len(forest["display_doc"]), 100)

    def test_missing_query_terms_do_not_mutate_shared_keyword_index(self):
        index = InvertedIndex(self.directory.name)
        index.build(self.documents)
        keys = set(index.index)
        self.assertEqual(index.bm25_search("unknownword"), [])
        self.assertEqual(set(index.index), keys)

    def test_an_empty_stopword_file_is_still_loaded_only_once(self):
        with patch("rag_search.search.tokenization.STOP_WORDS", None), patch("builtins.open", mock_open(read_data="")) as file:
            self.assertEqual(load_stopwords(), [])
            self.assertEqual(load_stopwords(), [])
            file.assert_called_once()
