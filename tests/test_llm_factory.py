import unittest
from unittest.mock import Mock, patch

from rag_search.app.bootstrap import build_runtime
from rag_search.config.settings import Settings
from rag_search.llm.client_factory import create_llm_client
from rag_search.search.query_enhancer import QueryEnhancer
from rag_search.search.reranker import IndividualLLMReranker
from tests.helpers import PROMPTS


class ClientFactoryTests(unittest.TestCase):
    def test_search_without_credentials_does_not_construct_an_sdk(self):
        with patch("rag_search.llm.openai_compatible_client.OpenAICompatibleClient") as sdk:
            self.assertIsNone(create_llm_client(Settings()))
            sdk.assert_not_called()

    def test_backend_is_selected_in_one_place(self):
        adapter = Mock()
        adapter.complete.side_effect = ["corrected query", '{"score": 8.2}']
        factory = Mock(return_value=adapter)
        settings = Settings(llm_backend="another_library", llm_api_key="test-key")
        with patch("rag_search.llm.client_factory.CLIENT_FACTORIES", {"another_library": factory}):
            client = create_llm_client(settings)
        factory.assert_called_once_with(settings)
        self.assertEqual(QueryEnhancer(client, prompts=PROMPTS).spell_correct("query"), "corrected query")
        self.assertEqual(IndividualLLMReranker(client, prompts=PROMPTS).rate_document("q", "t", "text"), 8.2)

    def test_unknown_backend_fails_before_loading_models(self):
        with patch("rag_search.app.bootstrap.build_movie_search") as build, self.assertRaisesRegex(
            ValueError, "Unsupported LLM_BACKEND"
        ):
            build_runtime(Settings(llm_backend="unknown"))
        build.assert_not_called()

    def test_a_backend_can_use_a_library_without_api_key_authentication(self):
        client = Mock()
        factory = Mock(return_value=client)
        settings = Settings(llm_backend="local_library")
        with patch("rag_search.llm.client_factory.CLIENT_FACTORIES", {"local_library": factory}):
            self.assertIs(create_llm_client(settings), client)

    def test_client_is_closed_if_index_initialization_fails(self):
        client = Mock()
        with patch("rag_search.app.bootstrap.create_llm_client", return_value=client), patch(
            "rag_search.app.bootstrap.build_movie_search", side_effect=RuntimeError("index failure")
        ), self.assertRaisesRegex(RuntimeError, "index failure"):
            build_runtime(Settings())
        client.close.assert_called_once()
