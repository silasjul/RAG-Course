import string
from nltk.stem import PorterStemmer

with open('./data/stopwords.txt', 'r') as stopwords_file:
    text = stopwords_file.read()
    stopwords = set(text.splitlines())

def text_processing(text: str) -> list:
    # Case sensitivity
    text = text.lower()

    # Punctuation - removes , . ! so on...
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Tokenization
    text = text.split(' ')
    text = [s for s in text if s] # remove empty tokens

    # Stop words - removes unneccesary stopwords
    text = set(text) - stopwords
    text = list(text)

    # Stemming - removing word variations (running, runs, ran -> run)
    stemmer = PorterStemmer()
    text = [stemmer.stem(s) for s in text]

    return text