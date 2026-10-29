"""Construct the movie application once; keep setup out of query execution."""

import hashlib
import json
from pathlib import Path

from rag_search.search.hybrid_search import HybridSearch
from rag_search.search.query_enhancer import QueryEnhancer
from rag_search.search.reranker import IndividualLLMReranker
from rag_search.search.retrievers import BM25Retriever, SemanticRetriever
from rag_search.app.runtime import SearchRuntime
from rag_search.search.search_service import SearchService
from rag_search.config.defaults import DEFAULT_CHUNK_OVERLAP, DEFAULT_SEMANTIC_CHUNK_SIZE
from rag_search.sources.movie_loader import load_movies
from rag_search.search.tokenization import load_stopwords
from rag_search.config.settings import Settings
from rag_search.config.prompt_profiles import load_prompt_profile
from rag_search.llm.client_factory import create_llm_client


def build_movie_search(settings: Settings, *, model=None) -> HybridSearch:
    # Import expensive libraries only when preparing a local collection.
    from rag_search.search.keyword_search import InvertedIndex
    from rag_search.search.semantic_search import ChunkedSemanticSearch

    documents = load_movies()
    if not documents:
        raise ValueError("The movie collection is empty")
    manifest = {
        "version": 1,
        "documents_sha256": hashlib.sha256(
            json.dumps(documents, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "stopwords_sha256": hashlib.sha256(
            "\n".join(load_stopwords()).encode("utf-8")
        ).hexdigest(),
        "embedding_model": settings.embedding_model,
        "chunker": "sentences-v1",
        "chunk_size": DEFAULT_SEMANTIC_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    }
    cache_dir = settings.movie_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    try:
        cached_manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        cached_manifest = None

    index = InvertedIndex(cache_dir=str(cache_dir))
    semantic = ChunkedSemanticSearch(
        settings.embedding_model, model=model, cache_dir=str(cache_dir)
    )
    required_files = (
        index.index_path, index.docmap_path, index.term_frequencies_path,
        index.doc_lengths_path, index.avg_doc_length_path,
        semantic.chunk_embeddings_path, semantic.chunk_metadata_path,
    )
    if cached_manifest == manifest and all(Path(path).is_file() for path in required_files):
        index.load()
        semantic.load_or_create_chunk_embeddings(documents)
    else:
        # A failed rebuild must not leave a manifest claiming the cache is complete.
        manifest_path.unlink(missing_ok=True)
        index.build(documents)
        index.save()
        semantic.build_chunk_embeddings(documents)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return HybridSearch(BM25Retriever(index), SemanticRetriever(semantic))


def build_runtime(settings: Settings | None = None, *, model=None) -> SearchRuntime:
    settings = settings if settings is not None else Settings.from_environment()
    collection = settings.collection("movies")
    prompts = load_prompt_profile(collection.prompt_profile, settings.prompts_dir)
    client = create_llm_client(settings)
    try:
        search = build_movie_search(settings, model=model)
        service = SearchService(
            search,
            enhancer=QueryEnhancer(client, prompts=prompts) if client else None,
            reranker=IndividualLLMReranker(client, prompts=prompts) if client else None,
        )
        return SearchRuntime({"movies": service}, close=client.close if client else None)
    except BaseException:
        if client is not None:
            client.close()
        raise
