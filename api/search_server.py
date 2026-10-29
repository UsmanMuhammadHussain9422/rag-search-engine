"""FastAPI application: HTTP contracts and one loaded runtime per ASGI process."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from api.schemas import ErrorResponse, HealthResponse, SearchRequest, SearchResponseSchema
from rag_search.app.bootstrap import build_runtime
from rag_search.app.runtime import CollectionNotFoundError, SearchRuntime
from rag_search.llm.llm_client import LLMError, LLMUnavailableError


def create_app(runtime_factory: Callable[[], SearchRuntime] = build_runtime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        runtime = await run_in_threadpool(runtime_factory)
        application.state.runtime = runtime
        try:
            yield
        finally:
            application.state.runtime = None
            await run_in_threadpool(runtime.close)

    application = FastAPI(title="RAG Search Engine", version="0.1.0", lifespan=lifespan)
    application.state.runtime = None

    @application.exception_handler(CollectionNotFoundError)
    async def collection_not_found(request: Request, exc: CollectionNotFoundError):
        return JSONResponse(status_code=404, content={"detail": "Collection not found"})

    @application.exception_handler(LLMUnavailableError)
    async def llm_unavailable(request: Request, exc: LLMUnavailableError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @application.exception_handler(LLMError)
    async def llm_failure(request: Request, exc: LLMError):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # Return stable error fields, without raw input or unserializable contexts.
        issues = [
            {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": issues})

    def get_runtime(request: Request) -> SearchRuntime:
        runtime = request.app.state.runtime
        if runtime is None:
            raise HTTPException(status_code=503, detail="Search is not ready")
        return runtime

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        return HealthResponse(status="ready", collections=get_runtime(request).collection_names)

    @application.post(
        "/search",
        response_model=SearchResponseSchema,
        response_model_exclude_none=True,
        responses={status: {"model": ErrorResponse} for status in (404, 422, 502, 503)},
    )
    def search(body: SearchRequest, request: Request) -> SearchResponseSchema:
        service = get_runtime(request).get_collection(body.collection)
        response = service.search(
            body.query, k=body.k, limit=body.limit,
            enhance=body.enhance, rerank_method=body.rerank_method,
        )
        return SearchResponseSchema.from_domain(body.collection, response)

    return application


# Importing this module creates routes, but does not read .env or load indexes.
app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the search API with Uvicorn")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("api.search_server:app", host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
