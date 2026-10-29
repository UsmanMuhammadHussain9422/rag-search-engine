from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import httpx
from pydantic import ValidationError

from api.schemas import SearchRequest, SearchResponseSchema
from api.search_client import SearchHTTPClient
from rag_search.search_models import SearchResponse
from tests.helpers import hit


def response_payload():
    return {"collection": "movies", **asdict(SearchResponse("bear", "bear", [hit()]))}


class APIContractTests(unittest.TestCase):
    def test_invalid_search_requests_are_rejected(self):
        invalid = [None, [], {}, {"query": ""}, {"query": "a", "extra": 1},
                   {"query": "a", "limit": True}, {"query": "a", "limit": 1.5},
                   {"query": "a", "k": float("inf")}, {"query": "a", "k": True},
                   {"query": "a", "collection": []}, {"query": "a", "enhance": []}]
        for body in invalid:
            with self.subTest(body=body), self.assertRaises(ValidationError):
                SearchRequest.model_validate(body)

    def test_response_round_trip_preserves_domain_results(self):
        payload = response_payload()
        result = SearchResponseSchema.model_validate(payload).to_domain()
        self.assertEqual(asdict(result), {key: value for key, value in payload.items() if key != "collection"})


class HTTPClientTests(unittest.TestCase):
    def test_request_and_response_use_typed_contracts(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=response_payload())
        with httpx.Client(base_url="http://testserver/", transport=httpx.MockTransport(handler)) as transport:
            with SearchHTTPClient("http://testserver", client=transport) as client:
                result = client.search("bear", limit=3)
            self.assertFalse(transport.is_closed)
        self.assertEqual(result.results[0]["id"], 1)
        self.assertEqual(requests[0].url.path, "/search")
        self.assertEqual(json.loads(requests[0].content)["limit"], 3)

    def test_http_errors_and_invalid_response_are_reported(self):
        for status, body, expected in ((404, {"detail": "Collection not found"}, "HTTP 404"),
                                       (200, {}, "invalid response")):
            with self.subTest(status=status), httpx.Client(
                base_url="http://testserver/",
                transport=httpx.MockTransport(lambda request: httpx.Response(status, json=body)),
            ) as transport:
                with self.assertRaisesRegex(RuntimeError, expected):
                    SearchHTTPClient("http://testserver", client=transport).search("bear")

    def test_owned_connection_pool_is_closed(self):
        transport = Mock()
        with patch("api.search_client.httpx.Client", return_value=transport):
            with SearchHTTPClient("http://testserver"):
                pass
        transport.close.assert_called_once()

    def test_remote_cli_does_not_import_embedding_models_or_llm_sdk(self):
        payload = repr(response_payload())
        script = (
            "import sys; from unittest.mock import patch; from cli import hybrid_search_cli; "
            "sys.argv = ['hybrid_search_cli'] + sys.argv[1:]; "
            "factory = patch('api.search_client.httpx.Client'); mock = factory.start(); "
            f"mock.return_value.post.return_value.json.return_value = {payload}; "
            "hybrid_search_cli.main(); mock.return_value.post.assert_called_once(); "
            "mock.return_value.close.assert_called_once(); "
            "assert 'rag_search.app.bootstrap' not in sys.modules; "
            "assert 'rag_search.search.semantic_search' not in sys.modules; "
            "assert 'rag_search.search.keyword_search' not in sys.modules; "
            "assert 'openai' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script, "rrf-search", "bear", "--server-url", "http://testserver"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Document 1", completed.stdout)
