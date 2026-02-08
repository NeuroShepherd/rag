

#!/usr/bin/env python3

import argparse
from semantic_search import verify_model, SemanticSearch

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # verify model parser
    verify_parser = subparsers.add_parser("verify", help="Verify that the semantic search model loads correctly")


    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()