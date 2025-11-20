import argparse

from lib.multimodal_search import verify_image_embedding, image_search_command


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify that image embedding works"
    )
    verify_parser.add_argument("image", type=str, help="Path to image")

    image_search_parser = subparsers.add_parser(
        "image_search",
        help="search for documents using image as an input image",
    )
    image_search_parser.add_argument("image", type=str, help="Path to image")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case "image_search":
            image_search_command(args.image)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
