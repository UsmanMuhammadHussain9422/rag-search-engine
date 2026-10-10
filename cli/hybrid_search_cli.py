import argparse
from lib.hybrid_search import normalise_command, weighted_search_command, rrf_search_command
from lib.search_utils import DEFAULT_SEARCH_LIMIT, DEFAULT_ALPHA, DEFAULT_K

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
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite"], help="Enhances the user query.")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalise_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit, args.enhance)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()