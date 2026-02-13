import argparse
from hybrid_search import normalize_scores_text, weighted_search_text, rrf_search_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # normalize subparser
    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", nargs="+", type=float, help="List of scores to normalize")

    # weighted search subparser
    weighted_parser = subparsers.add_parser("weighted-search", help="Perform a weighted hybrid search")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="Weighting factor for BM25 vs semantic search (0.0 to 1.0)")
    weighted_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    ## rrf-search parser
    rrf_parser = subparsers.add_parser("rrf-search", help="Perform a Reciprocal Rank Fusion (RRF) hybrid search")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF parameter k")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    rrf_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Method for reranking results after fusion")




    # parse the arguments
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_scores_text(*args.scores)
        case "weighted-search":
            weighted_search_text(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_text(args.query, args.k, args.limit, args.enhance, args.rerank_method)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()