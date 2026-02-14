
import argparse, mimetypes
from describe_image import describe_image_text


def main():
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to the image to describe")
    parser.add_argument("--query", type=str, required=True, help="Query to use for describing the image")


    

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"  # Default to JPEG if MIME type can't be determined

    with open(args.image, "rb") as f:
        image_content = f.read()

    describe_image_text(image_content = image_content, query = args.query, mime_type = mime)







if __name__ == "__main__":
    main()