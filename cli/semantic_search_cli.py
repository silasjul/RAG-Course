#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search,
    chunking_fixed,
    chunking_semantic,
    embed_chunks,
    search_chunked,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the sentence transformer model")

    search_parser = subparsers.add_parser(
        "embed_text", help="Embeds text using the sentence transformer model"
    )
    search_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser(
        "verify_embeddings",
        help="Verifies that the embeddings are build and correct. If it does not exist or is wrong it rebuilds the embeddings and caches them",
    )

    search_parser = subparsers.add_parser(
        "embed_query", help="Embeds query using the sentence transformer model"
    )
    search_parser.add_argument("query", type=str, help="query to embed")

    search_parser = subparsers.add_parser(
        "search", help="Search through the movies using semantic search"
    )
    search_parser.add_argument("query", type=str, help="query to search")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

    search_parser = subparsers.add_parser(
        "chunk",
        help="Split text into chunks of speficied sizes defaults to 200 chars per chunk",
    )
    search_parser.add_argument("text", type=str, help="text to split into chunks")
    search_parser.add_argument(
        "--chunk-sizes", default=200, type=int, help="chunk size"
    )
    search_parser.add_argument("--overlap", default=0, type=int, help="overlap size")

    search_parser = subparsers.add_parser(
        "semantic_chunk",
        help="Split text into sentences with specified sentence overlaps",
    )
    search_parser.add_argument("text", type=str, help="text to split into chunks")
    search_parser.add_argument(
        "--max-chunk-size",
        default=4,
        type=int,
        help="max amount of sentences in a chunk (default 4)",
    )
    search_parser.add_argument(
        "--overlap", default=0, type=int, help="amount of sentences to overlap"
    )

    subparsers.add_parser(
        "embed_chunks",
        help="chunks the moviedescriptions using semantic chunking, embeds them and caches the embeddings and metadata for fast document lookups",
    )

    search_parser = subparsers.add_parser(
        "search_chunked",
        help="Search through movies using chunked semantic embeddings",
    )
    search_parser.add_argument("query", type=str, help="query to search")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case "chunk":
            chunks = chunking_fixed(args.text, args.chunk_sizes, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for idx, c in enumerate(chunks, 1):
                print(f"{idx}. {c}")
        case "semantic_chunk":
            chunks = chunking_semantic(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for idx, c in enumerate(chunks, 1):
                print(f"{idx}. {c}")
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            search_chunked(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
