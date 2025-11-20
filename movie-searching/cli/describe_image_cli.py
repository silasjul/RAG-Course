import argparse
import mimetypes
from google.genai import types

from lib.gemini_utils import generate_response_parts


def main():
    parser = argparse.ArgumentParser(
        description="Convert image and text to a searchable query CLI"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image to combine with text query",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to ask llm",
    )

    args = parser.parse_args()
    image_path = args.image
    query = args.query

    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"

    with open(image_path, "rb") as f:
        image = f.read()

    system_prompt = """
    Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary"""

    parts = [
        system_prompt,
        types.Part.from_bytes(data=image, mime_type=mime),
        query.strip(),
    ]

    response = generate_response_parts(parts)

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
