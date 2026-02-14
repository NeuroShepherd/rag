

import argparse, json
from hybrid_search import HybridSearch, rag_text, rag_summary_text, rag_citations_text, rag_question_text


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    # summarize parser
    summarize_parser = subparsers.add_parser("summarize", help="Summarize a given text")
    summarize_parser.add_argument("query", type=str, help="Text to summarize")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Number of search results to use for summarization")

    # citation parser
    citation_parser = subparsers.add_parser("citations", help="Generate citations for a given query")
    citation_parser.add_argument("query", type=str, help="Search query to generate citations for")
    citation_parser.add_argument("--limit", type=int, default=5, help="Number of search results to use for citation generation")

    # question parser
    question_parser = subparsers.add_parser("question", help="Answer a question using RAG")
    question_parser.add_argument("query", type=str, help="Question to answer using RAG")
    question_parser.add_argument("--limit", type=int, default=5, help="Number of search results to use for answering the question")




    with open("data/movies.json", "r") as f:
        data = json.load(f)
    documents = data["movies"]


    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            search = HybridSearch(documents=documents)
            results = search.rrf_search(query=query, k=60, limit=5)
            rag_text(query, documents, results[:5])

        case "summarize":
            query = args.query
            search = HybridSearch(documents=documents)
            results = search.rrf_search(query=query, k=60, limit=args.limit)
            rag_summary_text(query, documents, results[:args.limit])
        case "citations":
            query = args.query
            search = HybridSearch(documents=documents)
            results = search.rrf_search(query=query, k=60, limit=args.limit)
            rag_citations_text(query, documents, results[:args.limit])
        case "question":
            query = args.query
            search = HybridSearch(documents=documents)
            results = search.rrf_search(query=query, k=60, limit=args.limit)
            rag_question_text(query, documents, results[:args.limit])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()