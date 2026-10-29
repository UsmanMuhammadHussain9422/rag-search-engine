"""Parse and dispatch hybrid search CLI commands."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from rag_search.config.defaults import DEFAULT_ALPHA, DEFAULT_K, DEFAULT_SEARCH_LIMIT
from rag_search.commands.hybrid_search_commands import (
    rrf_search_command,
    build_command,
    normalise_command,
    weighted_search_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="normalize embedded scores for bm25 and semantic scoring.")
    normalize_parser.add_argument("scores", type=float, nargs='+', help="Embedded scores to normalise")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Weighted-Search.")
    weighted_search_parser.add_argument("query", type=str, help="Search Query")
    weighted_search_parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Constant to control the weighting between bm25 and semantic scores")
    weighted_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Limit for the search results")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Reciprocal Rank Fusion-Search.")
    rrf_search_parser.add_argument("query", type=str, help="Search Query")
    rrf_search_parser.add_argument("--k", type=float, default=DEFAULT_K, help="Constant to controls how much more weight we give to higher-ranked results vs. lower-ranked ones")
    rrf_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Limit for the search results")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Enhances the user query.")
    rrf_search_parser.add_argument("--rerank-method", type=str, default=None, choices=["individual"], help="Reranks the results of a user query.")

    rrf_search_parser = subparsers.add_parser("build", help="Builds Files required for Search")

    args = parser.parse_args()

    match args.command:
        case "build":
            build_command()
        case "normalize":
            normalise_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit, args.enhance, args.rerank_method)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
