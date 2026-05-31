import argparse
from lib.hybrid_search import normalise_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    args = parser.parse_args()

    match args.command:
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()