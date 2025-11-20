import argparse

from lib.augmented_generation import (
    augmented_generation,
    summarize,
    citations,
    question,
)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    rag_summarize_parser = subparsers.add_parser(
        "summarize", help="Perform RAG (search + generate summary)"
    )
    rag_summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="How many documents to include in generation",
    )

    rag_citations_parser = subparsers.add_parser(
        "citations", help="Perform RAG (search + generate summary with citations)"
    )
    rag_citations_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="How many documents to include in generation",
    )

    rag_question_parser = subparsers.add_parser(
        "question",
        help="Perform RAG (search + generate answer to question about recieved documents)",
    )
    rag_question_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_question_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="How many documents to include in generation",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            result = augmented_generation(args.query)
            print("Search Results:")
            for doc in result["relevant_docs"]:
                print(f'    - {doc["document"]["title"]}')
            print("\nRAG Response:")
            print(result["llm_response"])
        case "summarize":
            result = summarize(args.query, args.limit)
            print("Search Results:")
            for doc in result["relevant_docs"]:
                print(f'    - {doc["document"]["title"]}')
            print("\nLLM Summary:")
            print(result["llm_response"])
        case "citations":
            result = citations(args.query, args.limit)
            print("Search Results:")
            for doc in result["relevant_docs"]:
                print(f'    - {doc["document"]["title"]}')
            print("\nLLM Answer:")
            print(result["llm_response"])
        case "question":
            result = question(args.query, args.limit)
            print("Search Results:")
            for doc in result["relevant_docs"]:
                print(f'    - {doc["document"]["title"]}')
            print("\nAnswer:")
            print(result["llm_response"])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
