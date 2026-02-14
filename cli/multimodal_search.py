

import os, json
from PIL import Image
from sentence_transformers import SentenceTransformer
from semantic_search import cosine_similarity



class MultimodalSearch():
    def __init__(self, documents=[], model_name: str = "clip-ViT-B-32") -> None:
        self.documents = documents
        self.texts = []
        for doc in self.documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")

        self.model = SentenceTransformer(model_name)
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path)
        embedding = self.model.encode([image])
        return embedding[0]
    
    def search_with_image(self, image_path, limit=5):
        image_embedding = self.embed_image(image_path)

        similarities = []
        for i, text_embedding in enumerate(self.text_embeddings):
            sim = cosine_similarity(image_embedding, text_embedding)
            similarities.append((i, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in similarities[:limit]:
            doc = self.documents[idx]
            results.append(
                dict(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:100],
                    score=score,
                )
            )

        return results

    

def verify_image_embedding(image_path):
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path, limit=5):
    with open("data/movies.json", "r") as f:
        data = json.load(f)
    searcher = MultimodalSearch(documents=data["movies"])
    results = searcher.search_with_image(image_path, limit=limit)

    return {
        "image_path": image_path,
        "results": results,
    }
    