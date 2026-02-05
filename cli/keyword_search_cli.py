#!/usr/bin/env python3

import argparse
from helpers import load_movies, load_stop_words, build_command, search, InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # search command
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    # build command
    subparsers.add_parser("build", help="Build the inverted index")

    # tf parser
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")



    args = parser.parse_args()

    movies = load_movies("data/movies.json")
    stop_words = load_stop_words("data/stopwords.txt")



    match args.command:
        case "build":
            build_command()
        case "search":
            # print the search query here
            search(args, movies, stop_words)
        case "tf":
            index = InvertedIndex()
            index.load()
            freq = index.get_tf(args.doc_id, args.term)
            if freq != 0:
                print(f"doc_id: {args.doc_id}, term: '{args.term}', frequency: {freq}")
            else:
                print(0)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()