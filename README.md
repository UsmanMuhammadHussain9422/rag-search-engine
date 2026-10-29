# RAG Search Engine

A small project for learning how search works and how Retrieval Augmented Generation (RAG) can improve search experiences.

## Purpose

This project explores the foundations of building a search system for a content-based application, using movie data as the dataset.

## Setup

1. Create and activate a virtual environment
2. Install dependencies with `uv`
3. Add the movie dataset to `data/movies.json`
4. Add the tokenizer stopword list to `data/stopwords.txt`

The dataset format is:

```json
{"movies": [{"id": 1, "title": "Example", "description": "The full movie description."}]}
```

## Keep search loaded in a server

From the project root, install the API dependencies, then start the server:

```bash
uv sync --extra api
uv run --locked --extra api python -m api.search_server
```

The installation step resolves dependencies and updates `uv.lock`. Leave the
server running while querying it from another terminal.

Startup loads the movie data, embedding model, BM25 index, and semantic index once.
If compatible indexes are missing, startup builds and saves them first. The default
embedding model may need to be downloaded on its first use. The API becomes ready after initialization finishes; `/health` reports its loaded collections.

In another terminal, send queries through the CLI:

```bash
uv run --locked python cli/hybrid_search_cli.py rrf-search "space adventure" --server-url http://127.0.0.1:8000
```

Remote mode sends HTTP requests without loading local indexes, the embedding model,
or an LLM SDK. Omitting `--server-url` keeps the existing local CLI mode, which
initializes resources for that invocation. The existing `normalize`, `build`, and
`weighted-search` commands remain available locally.

You can also call the API directly:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"collection":"movies","query":"space adventure","limit":5}'
```

`POST /search` accepts `collection` (currently `movies`), `query`, `limit` (1–100),
`k` (nonnegative RRF constant), optional `enhance` (`spell`, `rewrite`, or `expand`),
and optional `rerank_method` (`individual`). It returns the original query, effective
query, and results with full text, a display preview, scores, and rank metadata.

The central `api/` package contains the FastAPI application, its typed schemas,
and an HTTPX client with connection pooling. Uvicorn serves the ASGI application.
Interactive API documentation is at `http://127.0.0.1:8000/docs`, with the generated
schema at `/openapi.json`. Invalid request bodies return structured HTTP 422 errors.

The server binds to `127.0.0.1:8000`; `--host` and `--port` configure the listener.
Keep one server process to share one loaded model and collection. Every additional
worker process would have its own model and indexes. Restart after changing source
data or settings. Deployment outside local development needs its own environment
configuration; selecting FastAPI alone does not complete deployment setup.

## Optional LLM features

Plain hybrid search does not require an API key. Enhancement and individual
reranking use the configured LLM client. Existing `OPENROUTER_API_KEY` and
`LLM_MODEL` settings are supported. Configuration is read at application startup.

| Variable | Default / purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | Optional key for OpenRouter |
| `LLM_API_KEY` | Optional override for an OpenAI-compatible endpoint |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` |
| `LLM_MODEL` | `openrouter/free` |
| `LLM_BACKEND` | `openai_compatible`; selects a registered client factory |
| `LLM_MAX_TOKENS` | `1024`; tune for the selected model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` |
| `SEARCH_COLLECTIONS_CONFIG` | Optional path to collection settings; defaults to `rag_search/config/collections.toml` |
| `SEARCH_PROMPTS_DIR` | Optional profile directory; defaults to `rag_search/config/prompts/` |

Set these in the process environment or your local `.env`. Environment variables
take precedence. LLM calls are made only when a request enables enhancement or
reranking. Choose a model/endpoint supporting text output and JSON mode for
reranking; different model families may need adapter-specific request options.
Individual reranking makes one LLM call per candidate and can increase latency and cost.

```bash
uv run --locked python cli/hybrid_search_cli.py rrf-search "spcae adventure" \
  --server-url http://127.0.0.1:8000 --enhance spell --rerank-method individual
```

## Project layout

CLI files parse arguments and dispatch to command functions in `rag_search/commands/`.
All HTTP components live together under `api/`. Search algorithms and workflows
remain independent of the HTTP framework.

```text
cli/
  hybrid_search_cli.py       # argparse and dispatch
  keyword_search_cli.py
  semantic_search_cli.py
  test_llm.py               # Manual LLM smoke check
api/
  search_server.py          # FastAPI app factory, routes, and lifespan
  search_client.py          # Pooled HTTPX client
  schemas.py                # Shared HTTP request/response contracts
rag_search/
  app/
    bootstrap.py            # Construct dependencies and prepare indexes
    runtime.py              # Hold loaded collections and close clients
  commands/
    hybrid_search_commands.py
    keyword_search_commands.py
    semantic_search_commands.py
  config/
    settings.py
    defaults.py
    prompt_profiles.py
    collections.toml
    prompts/
      movies.toml
      documents.toml
  search/
    keyword_search.py
    semantic_search.py
    hybrid_search.py
    search_service.py
    query_enhancer.py
    reranker.py
    retrievers.py
    tokenization.py
  llm/
    llm_client.py           # Completion and lifecycle contracts
    client_factory.py       # Select and construct the configured adapter
    openai_compatible_client.py
  sources/
    movie_loader.py
  search_models.py          # Domain result structures, independent of HTTP
```

`SemanticRetriever` describes the retrieval operation. The collection and source
are selected during startup; passage-based book ingestion remains future work.

## Changing the LLM library

For an OpenAI-compatible endpoint, configure the model, endpoint, and credentials.
For a different SDK, add an adapter implementing the completion and cleanup
contracts, register its builder in `rag_search/llm/client_factory.py`, and select
its name with `LLM_BACKEND`. Existing query enhancement and reranking use that
adapter without importing its SDK. Only `openai_compatible` is registered by default.

See [the adapter guide](docs/llm_adapters.md) for the exact steps and contracts.

## Prompts selected by collection

A collection selects its domain instructions in [rag_search/config/collections.toml](rag_search/config/collections.toml):

```toml
[collections.movies]
prompt_profile = "movies"
```

That selects [rag_search/config/prompts/movies.toml](rag_search/config/prompts/movies.toml), which contains
separate system prompts for `enhancement.spell`, `enhancement.rewrite`,
`enhancement.expand`, and `reranking.individual`. The current detailed movie-scoring
instructions are preserved there. Edit this file to change movie behavior without
editing Python classes.

To add another policy, copy a profile to `rag_search/config/prompts/my_profile.toml`, customize
all four prompts, and select `prompt_profile = "my_profile"` for the collection.
[documents.toml](rag_search/config/prompts/documents.toml) provides a starting policy for searching
information inside documents. Selecting it changes LLM instructions; document
readers and passage indexing are still future work.

Profiles are loaded and validated once during startup, then injected into the
features. Restart the server after editing a profile or its selection. Invalid
profiles fail startup with a configuration error. Query requests choose a collection;
they do not supply system prompts or arbitrary profile paths.

Keep the response contracts when editing prompts: enhancement returns plain text,
expansion returns additional terms only, and individual reranking returns a JSON
object with a numeric `score` between 0 and 10. Runtime parsing and score validation
remain in Python regardless of the configured instructions.

## Development

Install the API extra to include the FastAPI integration tests, then run without
model downloads, real LLM calls, or reading your `.env`:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --locked --extra api python -m unittest discover -s tests -v
```

FastAPI integration tests use `TestClient` with fake search resources and run in
process without opening listening sockets. Client tests use HTTPX `MockTransport`.
If the API extra is absent, server integration tests are explicitly skipped while
core, schema, and client tests can still run with the core dependencies installed.

See [the architecture notes](docs/architecture.md) for component responsibilities,
cache behavior, and the next steps for searching inside books and documents.

## Notes

- `data/` is ignored by Git
- `.venv/` is ignored by Git

