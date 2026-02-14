

import argparse, json
from hybrid_search import HybridSearch, rag_text


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            with open("data/movies.json", "r") as f:
                data = json.load(f)
            documents = data["movies"]
            search = HybridSearch(documents=documents)
            results = search.rrf_search(query=query, k=60, limit=5)
            rag_text(query, documents, results[:5])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()