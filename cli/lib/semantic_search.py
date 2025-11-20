from sentence_transformers import SentenceTransformer
import numpy as np
import os
from lib.search_utils import CACHE_DIR, load_movies
import re
import json

from lib.search_utils import SCORE_PRECISION


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.embeddings = None
        self.documents = None
        self.document_map = {}

        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embedding(self, text):
        if len(text) == 0:
            raise ValueError("text must not be empty.")

        result = self.model.encode([text])
        return result[0]

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        query_embedding = self.generate_embedding(query)

        similarities = np.array(
            [cosine_similarity(query_embedding, emb) for emb in self.embeddings]
        )
        top_indices = np.argsort(similarities)[::-1][:limit]
        return [
            {
                "score": similarities[i],
                "title": self.documents[i]["title"],
                "description": self.documents[i]["description"],
            }
            for i in top_indices
        ]

    def build_embeddings(self, documents):
        self.documents = documents

        doc_strings = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_strings.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)
        self.save()
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            self.documents = documents
            for doc in documents:
                self.document_map[doc["id"]] = doc
            if len(self.documents) != len(self.embeddings):
                self.build_embeddings(documents)
        else:
            self.build_embeddings(documents)

        return self.embeddings

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

        self.chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        all_chunks = []
        chunk_metadata = []
        for doc_idx, doc in enumerate(documents):
            self.document_map[doc["id"]] = doc
            description = doc["description"]
            if not description or not description.strip():
                continue
            chunks = chunking_semantic(description, max_chunk_size=4, overlap=1)
            all_chunks.extend(chunks)
            for chunk_idx in range(len(chunks)):
                chunk_metadata.append(
                    {
                        "movie_idx": doc_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(chunks),
                    }
                )
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        self.save_chunk_embeddings()

    def save_chunk_embeddings(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump(
                {
                    "chunks": self.chunk_metadata,
                    "total_chunks": len(self.chunk_metadata),
                },
                f,
                indent=2,
            )

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(
            self.chunk_metadata_path
        ):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                data = json.load(f)
            self.chunk_metadata = data["chunks"]
        else:
            self.build_chunk_embeddings(documents)
        return self.chunk_embeddings

    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.model.encode(query.strip())
        chunk_scores = []
        for idx, c in enumerate(self.chunk_embeddings):
            c_sim = cosine_similarity(query_embedding, c)
            chunk_scores.append(
                {
                    "chunk_idx": idx,
                    "movie_idx": self.chunk_metadata[idx]["movie_idx"],
                    "score": c_sim,
                }
            )
        movie_scores = {}
        for c in chunk_scores:
            if (
                c["movie_idx"] not in movie_scores
                or movie_scores[c["movie_idx"]] < c["score"]
            ):
                movie_scores[c["movie_idx"]] = c["score"]
        results = sorted(movie_scores.items(), key=lambda key: key[1], reverse=True)[
            :limit
        ]
        return [
            {
                "id": self.documents[movie_id]["id"],
                "title": self.documents[movie_id]["title"],
                "document": self.documents[movie_id]["description"][:200],
                "score": round(score, SCORE_PRECISION),
                "metadata": {},
            }
            for movie_id, score in results
        ]


def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(ss.documents)}")
    print(
        f"Embeddings shape: {ss.embeddings.shape[0]} vectors in {ss.embeddings.shape[1]} dimensions"
    )


def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def search(query, limit=5):
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    results = ss.search(query, limit)
    for idx, result in enumerate(results, 1):
        print(
            f"{idx}. {result['title']} (score: {result['score']:.4f})\n{result['description'][:200]}...\n"
        )


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def chunking_fixed(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("Size must be positive")
    if overlap >= size:
        raise ValueError("Overlap must be less than size")
    text = text.strip()
    if not text:
        return []
    words = text.split()
    chunks = []
    step = size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i : i + size]
        chunks.append(" ".join(chunk_words))
    return chunks


def chunking_semantic(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    if max_chunk_size <= 0:
        raise ValueError("Size must be positive")
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    step = max_chunk_size - overlap
    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        stripped_chunk = " ".join(chunk_sentences).strip()
        if stripped_chunk:
            chunks.append(stripped_chunk)
    return chunks


def embed_chunks():
    css = ChunkedSemanticSearch()
    movies = load_movies()
    css.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(css.chunk_embeddings)} chunked embeddings")


def search_chunked(query, limit=5):
    css = ChunkedSemanticSearch()
    movies = load_movies()
    css.load_or_create_chunk_embeddings(movies)
    results = css.search_chunks(query, limit)
    for idx, result in enumerate(results, 1):
        print(f'{idx}. {result["title"]} (score: {result["score"]:.4f})')
        print(f"{result['document'][:200]}...\n")
