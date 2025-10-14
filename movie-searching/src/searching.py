from .text_processing_pipeline import text_processing
import json
from typing import Dict, List, TypedDict, Set
from collections import defaultdict
import pickle
from tqdm import tqdm
import os

class Movie(TypedDict):
    id: int
    title: str
    description: str

with open('./data/movies.json', 'r') as movie_file:
    data = json.load(movie_file)
    movies: List[Movie] = data['movies']

def search_movie(keyword) -> list:    
    results = []
    for idx, movie in enumerate(movies, start=1):
        # Process movie title
        title = movie["title"]
        title_processed = text_processing(title)

        # Process keyword
        keyword_processed = text_processing(keyword)

        # Convert to sets for fast checks
        common_elements = set(title_processed) & set(keyword_processed)

        # Is there a match
        if bool(common_elements):
          results.append(title)
          print(f'{idx}. {title}\n')
    
    if len(results) == 0:
       print("No results found")

    return results

class InvertedIndex():
    def __init__(self):
      self.index: Dict[str, Set[int]] = defaultdict(set) # Word -> List of document ids

      self.docmap: Dict[int, Movie] = dict() # ID -> Movie (for speedy lookups)

    def __add_document(self, doc_id: int, text: str):
      tokens = text_processing(text)
      for t in tokens:
         self.index[t].add(doc_id)
    
    def get_documents(self, token: str) -> List[Movie]:
      movie_ids = self.index[token]

      result = []
      for id in movie_ids:
         result.append(self.docmap[id])

      result.sort(key=lambda movie: movie['id'])
      return result

    def build(self):
      for m in tqdm(movies, desc="Building index and docmap"):
         # Add to index
         self.__add_document(m['id'], f"{m['title']} {m['description']}")

         # Add to docmap
         self.docmap[m['id']] = m

    def save(self):
      os.makedirs('cache', exist_ok=True)
      try:
        with open('cache/index.pkl', 'wb') as file:
            pickle.dump(self.index, file)
        print(f"Object successfully dumped index")
      except Exception as e:
          print(f"An error occurred dumping index: {e}")

      try:
        with open('cache/docmap.pkl', 'wb') as file:
            pickle.dump(self.docmap, file)
        print(f"Object successfully dumped docmap")
      except Exception as e:
          print(f"An error occurred dumping docmap: {e}")
