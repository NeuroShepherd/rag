

#!/usr/bin/env python3

import argparse
import json
from semantic_search import SemanticSearch, verify_model, embed_text, verify_embeddings, embed_query_text, chunk_text, semantic_chunking

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # verify model parser
    verify_parser = subparsers.add_parser("verify", help="Verify that the semantic search model loads correctly")

    # embed parser
    embed_parser = subparsers.add_parser("embed_text", help="Generate embedding for a given text")
    embed_parser.add_argument("text", type=str, help="Text to generate embedding for")
    
    # verify_embeddings parser
    verify_embed_parser = subparsers.add_parser("verify_embeddings", help="Verify that embeddings can be generated and loaded correctly")

    # embedquery parser
    embed_query_parser = subparsers.add_parser("embedquery", help="Generate embedding for a search query")
    embed_query_parser.add_argument("query", type=str, help="Search query to generate embedding for")

    # search parser
    search_parser = subparsers.add_parser("search", help="Search for movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    # chunking parser
    chunk_parser = subparsers.add_parser("chunk", help="Test text chunking functionality")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("-cs", "--chunk-size", type=int, default=200, help="Chunk size in characters")
    chunk_parser.add_argument("-o", "--overlap", type=int, default=0, help="Number of words to overlap between chunks")

    # semantic chunking parser
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Test semantic chunking functionality")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk semantically")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Maximum number of sentences per chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of sentences to overlap between chunks")



    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search = SemanticSearch()
            with open("data/movies.json", "r") as f:
                data = json.load(f)
            documents = data["movies"]
            search.load_or_create_embeddings(documents)
            results = search.search(args.query, args.limit)
            for i, result in enumerate(results):
                score, title, description = result["score"], result["title"], result["description"]
                print(f"{i}. {title} (score: {score:.4f})\n{description}\n")
        case "chunk":
            chunk_text(args.text, chunk_size=args.chunk_size, overlap=args.overlap)
        case "semantic_chunk":
            semantic_chunking(args.text, max_chunk_size=args.max_chunk_size, overlap=args.overlap)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()