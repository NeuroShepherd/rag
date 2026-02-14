

import argparse
from multimodal_search import verify_image_embedding


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # verify_image_embedding subparser
    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding", help="Verify that the image embedding model loads and generates embeddings correctly")
    verify_image_embedding_parser.add_argument("image", type=str, help="Path to the image to test embedding generation on")




    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()