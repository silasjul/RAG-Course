import argparse

from lib.hybrid_search import (
    normalize_scores,
    weighed_search,
    rrf_search,
    print_results,
)

from lib.gemini_utils import evaluate_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser(
        "normalize", help="Normalise search results to values ranging from 1-0"
    )
    normalize_parser.add_argument(
        "scores",
        type=float,
        nargs="+",
        help="a bunch of scores like: 1 6 8.2 9.1 4.0 1",
    )

    weighed_search_parser = subparser.add_parser(
        "weighted-search",
        help="weighted searching using an alpha value to weigh BM25 and semantic search results and combine then using normalized scores",
    )
    weighed_search_parser.add_argument("query", type=str, help="Your search query")
    weighed_search_parser.add_argument(
        "--alpha",
        default=0.5,
        type=float,
        help="1 being only keyword and 0 being only semantics (default 0.5)",
    )
    weighed_search_parser.add_argument(
        "--limit", type=int, default=5, help="how many results to show"
    )

    rrf_search_parser = subparser.add_parser(
        "rrf-search",
        help="combines searc h results using rankings instead of normalized values",
    )
    rrf_search_parser.add_argument("query", type=str, help="Your search query")
    rrf_search_parser.add_argument(
        "--k",
        default=60,
        type=int,
        help="how much scores drop off as they drop in rank. Low rank weights high scores a lot more and high k evens it out more",
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, default=5, help="how many results to show"
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Method to rerank the rrf results to provide better rankings",
    )
    rrf_search_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Will use an llm to evaluate the search results and give them a score",
    )
    rrf_search_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_scores(args.scores)
            for s in normalized_scores:
                print(f"* {s:.4f}")
        case "weighted-search":
            weighed_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            result = rrf_search(
                args.query,
                args.k,
                args.limit,
                args.enhance,
                args.rerank_method,
                args.debug,
            )
            print_results(result, args.rerank_method)
            if args.evaluate:
                evaluate_results(args.query, result)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
