from PIL import Image
import torch
from sentence_transformers import SentenceTransformer

from .search_utils import load_movies


class MultimodalSearch:
    def __init__(self, documents=[], model_name="clip-ViT-B-32"):
        self.model = self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        image_embeddings = self.model.encode([image])
        return image_embeddings[0]

    def search_with_image(self, image_path: str):
        embedding = self.embed_image(image_path)
        return self.search_with_embedding(embedding)

    def search_with_embedding(self, embedding):
        embedding_tensor = torch.tensor(embedding)
        text_embs_tensor = torch.tensor(self.text_embeddings)
        similarities = torch.cosine_similarity(
            embedding_tensor.unsqueeze(0), text_embs_tensor.squeeze(0)
        )
        top_k = min(5, len(self.documents))
        top_values, top_indices = torch.topk(similarities, top_k)
        results = []
        for i in range(top_k):
            idx = top_indices[i].item()
            doc = self.documents[idx]
            results.append(
                {
                    "document_id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "similarity_score": top_values[i].item(),
                }
            )
        return results


def verify_image_embedding(image_path: str):
    ms = MultimodalSearch()
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str):
    ms = MultimodalSearch(load_movies())
    result = ms.search_with_image(image_path)
    for idx, r in enumerate(result):
        print(f"{idx}.  {r['title']} (similarity: {r['similarity_score']:0.3f})")
        print(f"    {r['description'][:300]}...\n")
