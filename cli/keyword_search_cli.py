#!/usr/bin/env python3

import argparse
from helpers import load_movies, normalize_text, load_stop_words

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()






    movies = load_movies("data/movies.json")
    stop_words = load_stop_words("data/stopwords.txt")



    match args.command:
        case "search":
            # print the search query here
            counter = 1
            print(f"Searching for: {args.query}")
            # breakpoint()
            for movie in (movies):
                # breakpoint()
                query_words = normalize_text(args.query, stop_words=stop_words)
                title_words = normalize_text(movie["title"], stop_words=stop_words)
                if any(q in t for q in query_words for t in title_words):
                    print(f"{movie['id']}. Movie Title {movie['title']}")
                    counter += 1
                    if counter > 5:
                        break
                    
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()