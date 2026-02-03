#!/usr/bin/env python3

import argparse
import json
from icecream import ic


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    hello_parser = subparsers.add_parser("hello", help="Print hello message")
    hello_parser.add_argument("--name", type=str, help="Your name", default="User")

    args = parser.parse_args()

    movies = load_movies("data/movies.json")

    # breakpoint()

    match args.command:
        case "search":
            # print the search query here
            counter = 1
            print(f"Searching for: {args.query}")
            # ic(f"Searching for: {args.query}")
            for movie in (movies):
                if args.query.lower() in movie["title"].lower():
                    print(f"{movie['id']}. Movie Title {movie['title']}")
                    counter += 1
                    if counter > 5:
                        break
        case "hello":
            print("Hello! This is the Keyword Search CLI.")
            if args.name:
                print(f"Hello, {args.name}!")
        case _:
            parser.print_help()

            


def load_movies(file_path: str) -> dict:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["movies"]


if __name__ == "__main__":
    main()