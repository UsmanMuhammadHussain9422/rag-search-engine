import json
import unittest

from rag_search.search.hybrid_search import HybridSearch, normalise_scores
from rag_search.search.query_enhancer import QueryEnhancer
from rag_search.search.reranker import IndividualLLMReranker
from rag_search.search.search_service import SearchService
from rag_search.llm.llm_client import LLMError, LLMUnavailableError
from tests.helpers import PROMPTS, FakeLLM, FakeRetriever, hit


class SearchTests(unittest.TestCase):
    def test_rrf_combines_rankings_and_keeps_precision(self):
        keyword = FakeRetriever([hit(1), hit(2)])
        semantic = FakeRetriever([hit(2), hit(3)])
        results = HybridSearch(keyword, semantic).rrf_search("bear", limit=3)
        self.assertEqual([result["id"] for result in results], [2, 1, 3])
        self.assertAlmostEqual(results[0]["score"], 1 / 62 + 1 / 61)
        self.assertEqual(results[0]["metadata"]["keyword_rank"], 2)
        self.assertEqual(results[0]["metadata"]["semantic_rank"], 1)
        self.assertEqual(results[2]["metadata"]["keyword_score"], 0)

    def test_enhanced_query_reaches_both_retrievers_and_reranker(self):
        text = "A long movie description. " * 20
        keyword = FakeRetriever([hit(1, text)])
        semantic = FakeRetriever([hit(1, text)])
        enhancer_llm = FakeLLM(["bear movie"])
        rerank_llm = FakeLLM(['{"score": 8.5}'])
        service = SearchService(
            HybridSearch(keyword, semantic),
            enhancer=QueryEnhancer(enhancer_llm, prompts=PROMPTS),
            reranker=IndividualLLMReranker(rerank_llm, prompts=PROMPTS),
        )
        response = service.search("baer movie", enhance="spell", rerank_method="individual", limit=5)
        self.assertEqual(response.original_query, "baer movie")
        self.assertEqual(response.effective_query, "bear movie")
        self.assertEqual(keyword.calls, [("bear movie", 2500)])
        self.assertEqual(semantic.calls, [("bear movie", 2500)])
        payload = json.loads(rerank_llm.calls[0]["user"])
        self.assertEqual(payload["query"], "bear movie")
        self.assertEqual(payload["text"], text)
        self.assertEqual(response.results[0]["individual_score"], 8.5)
        self.assertEqual(enhancer_llm.calls[0]["output"], "text")
        self.assertEqual(rerank_llm.calls[0]["output"], "json")

    def test_plain_search_requires_no_llm(self):
        service = SearchService(HybridSearch(FakeRetriever([hit()]), FakeRetriever([])))
        self.assertEqual(len(service.search("bear").results), 1)
        with self.assertRaises(LLMUnavailableError):
            service.search("bear", enhance="spell")
        with self.assertRaises(LLMUnavailableError):
            service.search("bear", rerank_method="individual")

    def test_invalid_options_fail_before_retrieval(self):
        retriever = FakeRetriever([])
        service = SearchService(HybridSearch(retriever, retriever))
        for options in ({"limit": 0}, {"limit": 101}, {"k": -1}, {"k": float("nan")},
                        {"rerank_method": "batch"}, {"enhance": "unknown"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                service.search("bear", **options)
        self.assertEqual(retriever.calls, [])

    def test_expansion_preserves_the_original_query(self):
        enhancer = QueryEnhancer(FakeLLM(["grizzly forest"]), prompts=PROMPTS)
        self.assertEqual(enhancer.expand_query("bear"), "bear grizzly forest")

    def test_empty_enhancement_keeps_original_query(self):
        self.assertEqual(QueryEnhancer(FakeLLM([""]), prompts=PROMPTS).spell_correct("bear"), "bear")

    def test_reranking_does_not_mutate_input_and_breaks_ties_by_rrf(self):
        documents = [hit(1, score=0.0111), hit(2, score=0.0112), hit(3, score=0.5)]
        reranker = IndividualLLMReranker(FakeLLM([
            '{"score": 9}', '{"score": 9}', '{"score": 2}',
        ]), prompts=PROMPTS)
        self.assertEqual([r["id"] for r in reranker.rerank("bear", documents, 2)], [2, 1])
        self.assertNotIn("individual_score", documents[0])

    def test_invalid_llm_scores_are_normalized_to_llm_errors(self):
        for response in ("not JSON", "[]", "{}", '{"score": true}', '{"score": "8"}',
                         '{"score": -1}', '{"score": 11}', '{"score": NaN}', '{"score": Infinity}'):
            with self.subTest(response=response), self.assertRaises(LLMError):
                IndividualLLMReranker(FakeLLM([response]), prompts=PROMPTS).rate_document("q", "t", "text")

    def test_normalisation_handles_empty_and_equal_scores(self):
        self.assertEqual(normalise_scores([]), [])
        self.assertEqual(normalise_scores([4, 4]), [1, 1])
        self.assertEqual(normalise_scores([2, 4, 6]), [0, 0.5, 1])
