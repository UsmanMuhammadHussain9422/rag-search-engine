import argparse
from lib.search_utils import DEFAULT_SEARCH_LIMIT
from lib.keyword_search import search_command, tf_commmand, idf_command, tfidf_command, build_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="limit results")

    tf_parser = subparsers.add_parser("tf", help="Gets the term frequency for a given document ID and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a given term")
    idf_parser.add_argument("term", type=str, help="Term to get inverse document frequency for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF for a given document ID and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF for")

    build_parser = subparsers.add_parser("build", help="Build movies inverted index and save it to disk")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query, args.limit)
            for i, res in enumerate(results, 1):
                print(f'{i}. {res["title"]}')
        case "tf":
            print(f"Accessing TF for Doc_ID: {args.doc_id}")
            term_frequency = (tf_commmand(args.doc_id, args.term))
            print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {term_frequency}")
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()