
import os
import json
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types



def gemini_client():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    client = genai.Client(api_key=api_key)
    return client


def describe_image(image_content, query, client, mime_type):

    model = "gemini-2.5-flash"

    system_prompt = """
                    Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
                    - Synthesize visual and textual information
                    - Focus on movie-specific details (actors, scenes, style, etc.)
                    - Return only the rewritten query, without any additional commentary
                    """


    content = [
        system_prompt,
        types.Part.from_bytes(data=image_content, mime_type=mime_type),
        query.strip(),
    ]

    response = client.models.generate_content(
        model=model,
        contents=content
    )

    return response


def describe_image_text(image_content, query, mime_type):
    client = gemini_client()
    response = describe_image(image_content, query, client, mime_type)
    
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")
