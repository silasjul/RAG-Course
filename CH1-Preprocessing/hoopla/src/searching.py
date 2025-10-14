from .text_processing_pipeline import text_processing
import json

with open('./data/movies.json', 'r') as movie_file:
    data = json.load(movie_file)
    movies = data['movies']

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
    
    return results

