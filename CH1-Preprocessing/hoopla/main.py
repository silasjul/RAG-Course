from src.searching import search_movie
import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            search_query = args.query
            print("Searching for: " + search_query + "\n")
            search_movie(search_query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()