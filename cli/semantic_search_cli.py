#!/usr/bin/env python3

import argparse
from lib.semantic_search import search, verify_model, embed_text, verify_embeddings, embed_query_text, chunk_text
from lib.search_utils import DEFAULT_SEARCH_LIMIT, DEFAULT_CHUNK_SIZE

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings for the movie dataset")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a single text")
    embed_text_parser.add_argument("text", type=str, help="Text to Embed.")

    embed_query_parser = subparsers.add_parser("embed_query", help="Generate an embedding for the Query")
    embed_query_parser.add_argument("query", type=str, help="Query to Embed.")

    search_parser = subparsers.add_parser("search", help="Search")
    search_parser.add_argument("query", type=str, help="Query to Search.")
    search_parser.add_argument("--limit", "-l", type=int, default=DEFAULT_SEARCH_LIMIT, help="limit results")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", "-cs", type=int, default=DEFAULT_CHUNK_SIZE, help="Size of each chunk in words")
    chunk_parser.add_argument("--overlap", "-o",  nargs='?', type=int, default=DEFAULT_CHUNK_SIZE, help="Number of words to overlap between chunks")

    args = parser.parse_args()

    match args.command:
        case "search":
            search(args.query, args.limit)
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embed_query":
            embed_query_text(args.query)
        case "chunk":
          chunk_text(args.text, args.chunk_size, args.overlap)  
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()