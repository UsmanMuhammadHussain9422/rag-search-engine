import argparse
from lib.search_utils import DEFAULT_SEARCH_LIMIT
from lib.keyword_search import search_command, build_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="limit results")

    search_parser = subparsers.add_parser("build", help="Build movies inverted index and save it to disk")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query, args.limit)
            for i, res in enumerate(results, 1):
                print(f'{i}. {res["title"]}')
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()