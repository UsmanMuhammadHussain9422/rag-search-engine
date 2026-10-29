from concurrent.futures import ThreadPoolExecutor
import importlib.util
import unittest
from unittest.mock import Mock

from api.search_client import SearchHTTPClient
from rag_search.app.runtime import SearchRuntime
from rag_search.llm.llm_client import LLMError
from rag_search.search.hybrid_search import HybridSearch
from rag_search.search.query_enhancer import QueryEnhancer
from rag_search.search.search_service import SearchService
from tests.helpers import PROMPTS, FakeRetriever, hit

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None
if HAS_FASTAPI:
    from fastapi.testclient import TestClient
    from api.search_server import create_app


@unittest.skipUnless(HAS_FASTAPI, "Install the api extra to run FastAPI integration tests")
class APIServerTests(unittest.TestCase):
    def setUp(self):
        self.close = Mock()
        self.keyword = FakeRetriever([hit()])
        self.semantic = FakeRetriever([hit()])
        self.service = SearchService(HybridSearch(self.keyword, self.semantic))
        self.runtime = SearchRuntime({"movies": self.service}, close=self.close)
        self.factory = Mock(return_value=self.runtime)
        self.app = create_app(self.factory)

    def test_lifespan_loads_once_reuses_resources_and_closes(self):
        self.factory.assert_not_called()
        with TestClient(self.app) as client:
            for query in ("bear", "forest"):
                response = client.post("/search", json={"query": query})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["collection"], "movies")
            self.factory.assert_called_once_with()
            self.assertEqual(len(self.keyword.calls), 2)
            self.assertEqual(client.get("/health").json(), {"status": "ready", "collections": ["movies"]})
            self.close.assert_not_called()
        self.close.assert_called_once_with()
        self.assertIsNone(self.app.state.runtime)

    def test_invalid_requests_return_structured_422_errors(self):
        with TestClient(self.app) as client:
            for body in ({"query": " "}, {"query": "bear", "limit": 0},
                         {"query": "bear", "rerank_method": "batch"}, {"query": "bear", "extra": 1}):
                with self.subTest(body=body):
                    response = client.post("/search", json=body)
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertIsInstance(response.json()["detail"], list)
            for content in ('{', '{"query":"bear","k":NaN}'):
                response = client.post("/search", content=content, headers={"Content-Type": "application/json"})
                self.assertEqual(response.status_code, 422, response.text)

    def test_collection_and_unavailable_llm_errors(self):
        with TestClient(self.app) as client:
            response = client.post("/search", json={"query": "bear", "collection": "missing"})
            self.assertEqual(response.status_code, 404)
            response = client.post("/search", json={"query": "bear", "enhance": "spell"})
            self.assertEqual(response.status_code, 503)

    def test_provider_failure_returns_502(self):
        llm = Mock()
        llm.complete.side_effect = LLMError("The LLM request failed.")
        self.service.enhancer = QueryEnhancer(llm, prompts=PROMPTS)
        with TestClient(self.app) as client:
            response = client.post("/search", json={"query": "bear", "enhance": "spell"})
            self.assertEqual(response.status_code, 502)
            self.assertEqual(response.json()["detail"], "The LLM request failed.")

    def test_concurrent_queries_do_not_share_prompt_state(self):
        llm = Mock()
        llm.complete.side_effect = lambda **kwargs: "corrected " + kwargs["user"]
        self.service.enhancer = QueryEnhancer(llm, prompts=PROMPTS)
        queries = [f"query {index}" for index in range(8)]
        with TestClient(self.app) as client, ThreadPoolExecutor(max_workers=4) as executor:
            responses = list(executor.map(
                lambda query: client.post("/search", json={"query": query, "enhance": "spell"}), queries
            ))
        self.assertEqual([r.json()["effective_query"] for r in responses], ["corrected " + q for q in queries])

    def test_python_client_uses_the_same_contract(self):
        with TestClient(self.app) as client:
            with SearchHTTPClient("http://testserver", client=client) as search_client:
                response = search_client.search("bear")
                self.assertEqual(response.results[0]["id"], 1)
            self.assertEqual(client.get("/health").status_code, 200)

    def test_openapi_and_interactive_documentation(self):
        with TestClient(self.app) as client:
            schema = client.get("/openapi.json").json()
            operation = schema["paths"]["/search"]["post"]
            self.assertIn("SearchRequest", operation["requestBody"]["content"]["application/json"]["schema"]["$ref"])
            self.assertEqual(client.get("/docs").status_code, 200)

    def test_startup_failure_prevents_readiness(self):
        factory = Mock(side_effect=RuntimeError("index initialization failed"))
        with self.assertRaisesRegex(RuntimeError, "index initialization failed"):
            with TestClient(create_app(factory)):
                self.fail("The application must not start with unavailable indexes")
