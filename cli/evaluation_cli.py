import argparse
import json
from hybrid_search import HybridSearch


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

    with open("data/golden_dataset.json", "r") as f:
        golden_data = json.load(f)

    with open("data/movies.json", "r") as f:
        data = json.load(f)
    documents = data["movies"]



    # run evaluation logic here
    for case in golden_data["test_cases"]:
        search = HybridSearch(documents=documents)
        results = search.rrf_search(query = case["query"], k=60, limit=args.limit)
        # breakpoint()

        results = results[:args.limit]

        retrieved_movies = [result["title"] for result in results]
        relevant_movies = case["relevant_docs"]

        precision = len(set(retrieved_movies) & set(relevant_movies)) / len(retrieved_movies) if relevant_movies else 0
        recall = len(set(retrieved_movies) & set(relevant_movies)) / len(relevant_movies) if relevant_movies else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


        print(f"k={args.limit}\n")
        print(f"- Query: {case['query']}")
        print(f"    - Precision@{args.limit}: {precision:.4f}")
        print(f"    - Recall@{args.limit}: {recall:.4f}")
        print(f"    - F1 Score: {f1_score:.4f}")
        retrieved_str = ", ".join(retrieved_movies)
        relevant_str = ", ".join(relevant_movies)
        print(f"    - Retrieved: {retrieved_str}")
        print(f"    - Relevant: {relevant_str}")

if __name__ == "__main__":
    main()

