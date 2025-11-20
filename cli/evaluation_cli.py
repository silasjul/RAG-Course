import argparse
import json

from lib.search_utils import GOLDEN_DATASET_PATH
from lib.hybrid_search import rrf_search


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # evaluation logic

    with open(GOLDEN_DATASET_PATH, "r") as f:
        golden_dataset = json.load(f)

    print(f"k={limit}\n")

    for test_case in golden_dataset["test_cases"]:
        query, relevant_docs = test_case.values()
        result = rrf_search(query, k=60, limit=limit)

        # check how many correct results was found
        correct_count = 0
        retrieved_titles = []
        for r in result:
            doc_title = r["document"]["title"]
            retrieved_titles.append(doc_title)
            if doc_title in relevant_docs:
                correct_count += 1

        precision = correct_count / len(retrieved_titles)
        recall = correct_count / len(relevant_docs)
        f1 = 2 * (precision * recall) / (precision + recall)

        print(f"- Query: {query}")
        print(f"    - Precision@{limit}: {precision:0.4f}")
        print(f"    - Recall@{limit}: {recall:0.4f}")
        print(f"    - F1 Score: {f1:0.4f}")
        print(f"    - Retrieved: {', '.join(retrieved_titles)}")
        print(f"    - Relevant: {', '.join(relevant_docs)}\n")


if __name__ == "__main__":
    main()
