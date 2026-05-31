import argparse
from lib.hybrid_search import normalise_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    normalize_parser = subparsers.add_parser("normalize", help="normalize embedded scores for bm25 and semantic scoring.")
    normalize_parser.add_argument("scores", type=float, nargs='+', help="Search query")

    args = parser.parse_args()

    match args.command:
        case "normalize":
          normalise_command(args.scores)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()