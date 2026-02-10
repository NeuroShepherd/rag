
from sentence_transformers import SentenceTransformer
from torch import embedding
import numpy as np
import os, json, re

class SemanticSearch():
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if len(text.strip()) == 0:
            raise ValueError("Input text cannot be empty or whitespace.")
        
        embedding = self.model.encode([text])
        return embedding[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc
        
        string_rep = []
        for doc in documents:
            title = doc.get('title', '')
            description = doc.get('description', '')
            combined = f"{title}: {description}"
            string_rep.append(combined)
        
        self.embeddings = self.model.encode(string_rep, show_progress_bar=True)

        with open("cache/movie_embeddings.npy", "wb") as f:
            np.save(f, self.embeddings)
        
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc

        if os.path.exists("cache/movie_embeddings.npy"):
            with open("cache/movie_embeddings.npy", "rb") as f:
                self.embeddings = np.load(f)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        else:
            return self.build_embeddings(documents)
        
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        # Compute cosine similarity between query and all documents
        cosine_scores = np.dot(self.embeddings, query_embedding) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding))
        # Create a list of (similarity_score, document) tuples.
        scored_documents = [
            (cosine_scores[i], self.documents[i]) for i in range(len(self.documents))
        ]
        # Sort the documents by similarity score in descending order.
        scored_documents.sort(key=lambda x: x[0], reverse=True)
        # Return the top results (up to limit) as a list of dictionaries, each containing: score, title, description
        return [
            {
                "score": score,
                "title": doc.get("title", ""),
                "description": doc.get("description", "")
            }
            for score, doc in scored_documents[:limit]
        ]


def verify_model():
    search = SemanticSearch()
    # breakpoint()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def embed_text(text):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    search = SemanticSearch()
    # load movies.json into a list
    with open("data/movies.json", "r") as f:
        data = json.load(f)
    documents = data["movies"]
    embeddings = search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def chunk_text(text, chunk_size=200, overlap=0):
    words = re.split(r"(?<=[.!?])\s+", text)
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

    # To get strings instead of lists:
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")

def semantic_chunking(text, max_chunk_size=4, overlap=0):
    # Split on '. ' but keep the period with each sentence
    sentences = re.split(r'(?<=\.)\s+', text)
    chunks = [sentences[i:i + max_chunk_size] for i in range(0, len(sentences), max_chunk_size)]

    # To get strings instead of lists:
    chunks = [' '.join(sentences[i:i + max_chunk_size]) for i in range(0, len(sentences), max_chunk_size - overlap)]

    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")