

import os
from PIL import Image
from sentence_transformers import SentenceTransformer



class MultimodalSearch():
    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path)
        embedding = self.model.encode([image])
        return embedding[0]
    

def verify_image_embedding(image_path):
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")