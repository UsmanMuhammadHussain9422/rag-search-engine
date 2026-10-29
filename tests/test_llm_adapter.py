from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from openai import OpenAIError

from rag_search.llm.llm_client import LLMError
from rag_search.llm.openai_compatible_client import OpenAICompatibleClient


def completion(content="answer", finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish_reason,
    )])


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.sdk = Mock()
        self.sdk.chat.completions.create.return_value = completion()
        self.client = OpenAICompatibleClient(api_key="test-key", model="test-model", client=self.sdk)

    def test_plain_text_and_json_use_different_output_modes(self):
        for output, expected in (("text", "text"), ("json", "json_object")):
            self.assertEqual(self.client.complete(system="JSON if requested", user="question", output=output), "answer")
            arguments = self.sdk.chat.completions.create.call_args.kwargs
            self.assertEqual(arguments["response_format"], {"type": expected})
            self.assertEqual(arguments["model"], "test-model")

    def test_empty_missing_and_incomplete_responses_fail(self):
        for response in (completion(None), completion(""), completion("partial", "length"),
                         completion(None, "content_filter"), SimpleNamespace(choices=[])):
            self.sdk.chat.completions.create.return_value = response
            with self.subTest(response=response), self.assertRaises(LLMError):
                self.client.complete(system="s", user="u")

    def test_transport_error_does_not_expose_provider_details(self):
        self.sdk.chat.completions.create.side_effect = OpenAIError("sensitive provider detail")
        with self.assertRaises(LLMError) as raised:
            self.client.complete(system="s", user="u")
        self.assertNotIn("sensitive", str(raised.exception))

    def test_client_can_be_closed(self):
        self.client.close()
        self.sdk.close.assert_called_once()
