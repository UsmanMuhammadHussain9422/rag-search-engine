"""Parse and dispatch keyword search CLI commands."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from rag_search.config.defaults import DEFAULT_SEARCH_LIMIT, BM25_K1, BM25_B
from rag_search.commands.keyword_search_commands import (
    search_command,
    bm25_search_command,
    tf_commmand,
    idf_command,
    bm25_idf_command,
    tfidf_command,
    bm25_tf_command,
    build_command,
)


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

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 B parameter")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", "-l", type=int, default=DEFAULT_SEARCH_LIMIT, help="limit results")
    bm25search_parser.add_argument("--description", "-d", type=bool, default=False, help="Show full movie descriptions")


    build_parser = subparsers.add_parser("build", help="Build movies inverted index and save it to disk")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query, args.limit)
            if len(results) == 0:
                print(f"Sorry No Results Found for the Query: {args.query}")
                return
            for i, res in enumerate(results, 1):
                print(f'{i}. {res["title"]}')
        case "bm25search":
            print(f"Searching for: {args.query}")
            results = bm25_search_command(args.query, args.limit)
            if len(results) == 0:
                print(f"Sorry No Results Found for the Query: {args.query}")
                return
            for i, res in enumerate(results, 1):
                print(f'{i}. ({res["id"]}){res["title"]} - Score: {res["score"]:.2f}')
                if args.description:
                    print(res["document"])
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
        case "bm25idf":
            bm25_idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "bm25tf":
            print(f"Accessing BM25-TF for Doc_ID: {args.doc_id}")
            bm25tf = (bm25_tf_command(args.doc_id, args.term, args.k1, args.b))
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
