#!/usr/bin/env python3

import argparse
from helpers import load_movies, load_stop_words, build_command, search, InvertedIndex, BM25_K1, BM25_B

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

    # inverse document frequency parser
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF")

    # term freuquency - inverse document frequency parser (TFIDF)
    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for")

    # bm25idf parser
    bm25idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a term")
    bm25idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    
    # bm25tf parser
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")


    args = parser.parse_args()

    movies = load_movies("data/movies.json")
    stop_words = load_stop_words("data/stopwords.txt")



    match args.command:
        case "build":
            build_command()
        case "search":
            print(f"Searching for: {args.query}")
    
            # Load the inverted index
            index = InvertedIndex()
            index.load()
            # print the search query here
            search(index, args, movies, stop_words)
        case "tf":
            index = InvertedIndex()
            index.load()
            freq = index.get_tf(args.doc_id, args.term)
            if freq != 0:
                print(f"doc_id: {args.doc_id}, term: '{args.term}', frequency: {freq}")
            else:
                print(0)
        case "idf":
            index = InvertedIndex()
            index.load()
            idf_value = index.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf_value:.2f}")
        case "tfidf":
            index = InvertedIndex()
            index.load()
            tfidf_value = index.get_tfidf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf_value:.2f}")
        case "bm25idf":
            index = InvertedIndex()
            index.load()
            bm25idf_value = index.get_bm25idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf_value:.2f}")
        case "bm25tf":
            index = InvertedIndex()
            index.load()
            bm25tf_value = index.get_bm25_tf(args.doc_id, args.term, k1=args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf_value:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()