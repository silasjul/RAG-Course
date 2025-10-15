import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.searching import MovieSearcher, InvertedIndex

def main() -> None:
  parser = argparse.ArgumentParser(description="Keyword Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")

  # Search command
  search_parser = subparsers.add_parser("search", help="Search movies using BM25")
  search_parser.add_argument("query", type=str, help="Search query")

  # Build command
  search_parser = subparsers.add_parser("build", help="Build indexing for super fast searching! (build required before searching)")

  # tf command
  search_parser = subparsers.add_parser("tf", help="Check the amount of times a word appeared in a movies title and description")
  search_parser.add_argument("id", type=int, help="Movie id")
  search_parser.add_argument("word", type=str, help="Word")

  # idf command
  search_parser = subparsers.add_parser("idf", help="Check the inverse document frequency to see the amount of times a word appeared across all movies titles and descriptions")
  search_parser.add_argument("word", type=str, help="Word")

  args = parser.parse_args()

  match args.command:
    case "search":
      search_query = args.query
      print("Searching for: " + search_query + "\n")

      ms = MovieSearcher()
      results = ms.search_movie(search_query)
      show_amount = 5
      for idx, movie in enumerate(results):
        print(f"{movie['id']}. {movie['title']}")
        if idx >= show_amount-1:
          break


    case "build":
      ii = InvertedIndex()
      ii.build()
      ii.save()

    case "tf":
      word = args.word
      id = args.id
      print(f'Frequencies for word: "{word}" In movie with id: {id}')

      ii = InvertedIndex()
      ii.load()
      tf = ii.get_tf(id, word)
      print(f"TF: {tf}")
    
    case "idf":
      word = args.word
      print(f'Inverse document frequency for word: "{word}"')

      ii = InvertedIndex()
      ii.load()
      idf = ii.calculate_idf(word)
      print(f"IDF: {idf}")

    case _:
      parser.print_help()


if __name__ == "__main__":
  main()