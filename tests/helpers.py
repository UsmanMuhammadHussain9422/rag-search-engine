from rag_search.config.prompt_profiles import PromptProfile

PROMPTS = PromptProfile(
    spell="Fix spelling; return text.",
    rewrite="Rewrite the query; return text.",
    expand="Return additional terms only.",
    rerank='Return JSON with a numeric score from 0 to 10: {"score": 7.4}',
)

from rag_search.search_models import format_search_result


def hit(identifier=1, text="A bear travels through the forest.", score=0.8):
    return format_search_result(identifier, f"Document {identifier}", text, score)


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, limit):
        self.calls.append((query, limit))
        return self.results[:limit]


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)
