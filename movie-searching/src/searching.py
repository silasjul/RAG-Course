from .text_processing_pipeline import text_processing
import json
from typing import Dict, List, TypedDict, Set, Counter
from collections import defaultdict, Counter
import pickle
from tqdm import tqdm
import math
import os

class Movie(TypedDict):
  id: int
  title: str
  description: str

class MovieSearcher():
  def __init__(self):
    self.ii = InvertedIndex()
    self.ii.load()

  def search_movie(self, query) -> List[Movie]:  
    search_tokens = text_processing(query)
    return self.ii.get_documents(search_tokens, amount=9999)


class InvertedIndex():
  def __init__(self):
    self.index: Dict[str, Set[int]] = defaultdict(set) # Word -> List of document ids

    self.docmap: Dict[int, Movie] = dict() # ID -> Movie (for speedy lookups)

    self.term_frequencies: Dict[int, Counter] = defaultdict(Counter) # ID -> Counter for token frequencies

  def __add_document(self, doc_id: int, text: str):
    tokens = text_processing(text)
    token_counts = Counter(tokens)
    for t, count in token_counts.items():
      self.index[t].add(doc_id)
      self.term_frequencies[doc_id][t] = count
  
  def get_documents(self, token: str) -> List[Movie]:
    movie_ids = self.index[token]

    return self.__docs_by_ids(movie_ids)
  
  def get_documents(self, tokens: List[str], amount = 5) -> List[Movie]:
    # Find all movie ids
    all_movie_ids = set()
    for t in tokens:
      ids = self.index[t]

      if len(all_movie_ids) >= amount:
        break

      for id in ids:
        all_movie_ids.add(id)

        if len(all_movie_ids) >= amount:
          break
    
    return self.__docs_by_ids(all_movie_ids)

  def __docs_by_ids(self, ids: Set[int]):
    # Find all movies
    result = []
    for id in ids:
      result.append(self.docmap[id])

    result.sort(key=lambda movie: movie['id'])
    return result
  
  def get_tf(self, doc_id: int, token: str) -> int:
    return self.term_frequencies[doc_id][token]
  
  def calculate_idf(self, token: str) -> int:
    token = text_processing(token)[0]

    movies_count = len(self.docmap)
    token_count = len(self.index[token])

    return math.log((movies_count + 1) / (token_count + 1)) # adding 1 to prevent dividing with 0

  def build(self):
    with open('./data/movies.json', 'r') as movie_file:
      data = json.load(movie_file)
      movies: List[Movie] = data['movies']

    for m in tqdm(movies, desc="Building"):
      # Add to index
      self.__add_document(m['id'], f"{m['title']} {m['description']}")

      # Add to docmap
      self.docmap[m['id']] = m

  def save(self):
    os.makedirs('cache', exist_ok=True)
    try:
      with open('cache/index.pkl', 'wb') as file:
          pickle.dump(self.index, file)
      print(f"index successfully dumped")
    except Exception as e:
      print(f"An error occurred dumping index: {e}")

    try:
      with open('cache/docmap.pkl', 'wb') as file:
          pickle.dump(self.docmap, file)
      print(f"docmap successfully dumped")
    except Exception as e:
      print(f"An error occurred dumping docmap: {e}")
    
    try:
      with open('cache/term_frequencies.pkl', 'wb') as file:
          pickle.dump(self.term_frequencies, file)
      print(f"term_frequencies successfully dumped")
    except Exception as e:
      print(f"An error occurred dumping term_frequencies: {e}")

  def load(self):
    try:
      with open('./cache/index.pkl', 'rb') as index_file:
        self.index = pickle.load(index_file)
      with open('./cache/docmap.pkl', 'rb') as docmap_file:
        self.docmap = pickle.load(docmap_file)
      with open('./cache/term_frequencies.pkl', 'rb') as term_frequencies_file:
        self.term_frequencies = pickle.load(term_frequencies_file)
    except Exception as e:
      print(f"An error occurred during loading: {e}")
      raise Exception("builds didn't load properly.")
