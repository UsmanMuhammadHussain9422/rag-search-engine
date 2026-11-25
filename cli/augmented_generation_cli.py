import argparse
from lib.search_utils import DEFAULT_K, DEFAULT_SEARCH_LIMIT
from lib.augmented_search import augmented_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument("--k", type=float, default=DEFAULT_K, help="Constant to controls how much more weight we give to higher-ranked results vs. lower-ranked ones")
    rag_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Limit for the search results")
    rag_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Enhances the user query.")
    rag_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Re-ranking method")
    rag_parser.add_argument("--verbose", action="store_true", help="Prints out results on the console")

    args = parser.parse_args()

    match args.command:
        case "rag":
            response_text = augmented_search_command(args.query, k=args.k, limit=args.limit, enhance_type=args.enhance, rerank_method=args.rerank_method, verbose=args.verbose)
            print(response_text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()