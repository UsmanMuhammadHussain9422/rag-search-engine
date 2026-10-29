# Changing the LLM model, provider, or library

These are three different changes:

| Change | Where to make it |
| --- | --- |
| Another model on the same endpoint | `LLM_MODEL` |
| Another compatible endpoint using the existing SDK | `LLM_BASE_URL`, model, and credentials |
| A different SDK or local inference library | New adapter plus a registered builder |

The default `LLM_BACKEND` is `openai_compatible`. This selects the existing adapter
using the OpenAI SDK; changing this value does not install or implement another SDK.

## Adapter contract

`rag_search/llm/llm_client.py` defines two small contracts. Search features need:

```python
def complete(
    self,
    *,
    system: str,
    user: str,
    output: Literal["text", "json"] = "text",
) -> str:
    ...
```

The application owns the adapter's lifetime, so adapters returned by the factory
also implement `close()`. A library with no resources to close can use a no-op.

The adapter translates the system/user strings into its library's input format,
selects text or JSON output, and extracts a response string. Keep SDK objects
inside the adapter. Normalize transport failures and unusable/incomplete responses
to `LLMError`, with a safe public message. Domain JSON parsing and the 0–10 score
validation remain in the reranker.

If a chosen model/library cannot supply the required JSON response, report that
limitation rather than silently changing the scoring contract.

## Extension steps

1. Add the SDK dependency using the project dependency tooling.
2. Create `rag_search/llm/your_library_client.py` implementing the methods above.
3. Add a builder function in `rag_search/llm/client_factory.py`. It receives a
   `Settings` object and constructs your adapter. Add provider-specific settings
   when needed; do not read environment variables throughout feature classes.
4. Register that builder in `CLIENT_FACTORIES` under a descriptive backend name.
5. Set `LLM_BACKEND` to that registered name and configure the model/credentials.
6. Restart the application and test the adapter against representative queries.

Builders decide their backend's authentication rules. The default builder returns
`None` without a configured API key, preserving plain search with no LLM client.
A local-library builder can construct its adapter without any API key. An optional
backend may return `None` when it is not configured.

There is no dynamic plugin discovery. The explicit registry keeps supported
backends visible in one file. You do not need to edit query enhancement, reranking,
search service, CLI dispatch, or API routes when adding another SDK adapter.

## Tests to add for an adapter

- Input translation, text extraction, and JSON output selection using a fake SDK.
- Empty, incomplete, malformed, and transport-error responses.
- Client cleanup, including startup failure.
- Enhancement and reranking using the adapter without SDK-specific feature code.

The existing adapter tests and `tests/test_llm_factory.py` show these boundaries.
Live model evaluation is separate from these deterministic unit tests.
