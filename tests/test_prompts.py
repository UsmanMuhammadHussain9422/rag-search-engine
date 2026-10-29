from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from rag_search.app.bootstrap import build_runtime
from rag_search.config.prompt_profiles import PromptProfile, load_prompt_profile
from rag_search.config.settings import Settings
from rag_search.search.query_enhancer import QueryEnhancer
from rag_search.search.hybrid_search import HybridSearch
from rag_search.search.reranker import IndividualLLMReranker
from tests.helpers import FakeLLM, FakeRetriever, hit


class PromptConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.collections = Path(self.directory) / "collections.toml"
        self.collections.write_text('[collections.movies]\nprompt_profile = "documents"\n')
        self.settings = replace(Settings(), collections_file=self.collections)

    def test_profiles_have_different_domain_instructions(self):
        movies = load_prompt_profile("movies", self.settings.prompts_dir)
        documents = load_prompt_profile("documents", self.settings.prompts_dir)
        self.assertIn("movie relevance scoring", movies.rerank)
        self.assertIn("document passage", documents.rerank)
        self.assertNotEqual(movies.rewrite, documents.rewrite)
        self.assertIn('{"score": 7.4}', movies.rerank)
        self.assertIn('{"score": 7.4}', documents.rerank)

    def test_feature_classes_use_injected_prompts_verbatim(self):
        profile = PromptProfile(
            spell="Custom spelling instructions {literal braces}",
            rewrite="Custom rewriting instructions",
            expand="Custom expansion instructions",
            rerank='Custom JSON instructions: {"score": 7.4}',
        )
        llm = FakeLLM(["fixed", "rewritten", "extra terms", '{"score": 7.4}'])
        enhancer = QueryEnhancer(llm, prompts=profile)
        enhancer.spell_correct("query")
        enhancer.rewrite_query("query")
        enhancer.expand_query("query")
        IndividualLLMReranker(llm, prompts=profile).rate_document("query", "title", "text")
        self.assertEqual([call["system"] for call in llm.calls], [
            profile.spell, profile.rewrite, profile.expand, profile.rerank,
        ])

    def test_collection_configuration_selects_profile_at_startup(self):
        llm = Mock()
        llm.complete.side_effect = ["effective query", '{"score": 8.5}']
        search = HybridSearch(FakeRetriever([hit()]), FakeRetriever([]))
        with patch("rag_search.app.bootstrap.build_movie_search", return_value=search), patch(
            "rag_search.llm.openai_compatible_client.OpenAICompatibleClient", return_value=llm
        ):
            runtime = build_runtime(replace(self.settings, llm_api_key="test-key"))
        self.addCleanup(runtime.close)
        documents = load_prompt_profile("documents", self.settings.prompts_dir)
        service = runtime.get_collection("movies")
        with patch.object(Path, "open", side_effect=AssertionError("Unexpected configuration reload")):
            response = service.search("query", limit=1, enhance="spell", rerank_method="individual")
        self.assertEqual(response.effective_query, "effective query")
        self.assertEqual(llm.complete.call_args_list[0].kwargs["system"], documents.spell)
        self.assertEqual(llm.complete.call_args_list[1].kwargs["system"], documents.rerank)
        runtime.close()
        llm.close.assert_called_once()

    def test_unknown_profile_fails_before_loading_indexes(self):
        self.collections.write_text('[collections.movies]\nprompt_profile = "missing"\n')
        with patch("rag_search.app.bootstrap.build_movie_search") as build, self.assertRaisesRegex(
            ValueError, "Could not load prompt profile"
        ):
            build_runtime(self.settings)
        build.assert_not_called()

    def test_missing_or_empty_prompt_is_rejected(self):
        directory = Path(self.directory)
        for content in (
            '[enhancement]\nspell = "fix"\n',
            '[enhancement]\nspell = ""\nrewrite = "rewrite"\nexpand = "expand"\n'
            '[reranking]\nindividual = "score"\n',
            'enhancement = 12\n',
        ):
            with self.subTest(content=content):
                (directory / "custom.toml").write_text(content)
                with self.assertRaisesRegex(ValueError, "requires nonblank text"):
                    load_prompt_profile("custom", directory)

    def test_profile_name_cannot_escape_its_directory(self):
        for name in ("../private", "/absolute", "", "nested/profile"):
            with self.subTest(name=name), patch.object(Path, "open") as read, self.assertRaises(ValueError):
                load_prompt_profile(name, Path(self.directory))
            read.assert_not_called()

    def test_missing_collection_policy_is_rejected(self):
        for content in ('[collections.other]\nprompt_profile = "movies"\n',
                        '[collections.movies]\nprompt_profile = ""\n', 'collections = 3\n'):
            self.collections.write_text(content)
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, "requires a prompt_profile"):
                self.settings.collection("movies")

    def test_environment_can_override_configuration_locations_without_reading_dotenv(self):
        environment = {
            "SEARCH_COLLECTIONS_CONFIG": str(self.collections),
            "SEARCH_PROMPTS_DIR": self.directory,
        }
        with patch.dict(os.environ, environment, clear=True), patch(
            "rag_search.config.settings.load_dotenv"
        ):
            settings = Settings.from_environment()
        self.assertEqual(settings.collections_file, self.collections)
        self.assertEqual(settings.prompts_dir, Path(self.directory))
