
from sentence_transformers import SentenceTransformer
from torch import embedding
import numpy as np
import os, json, re

class SemanticSearch():
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
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
    








class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}

        chunk_strings = []
        chunk_metadata = []

        for doc in documents:
            if not doc.get('description'):
                continue
            # use semantic chunking to split the description into 4 sentence chunks with 1-sentence overlap
            chunks = semantic_chunking(doc['description'], max_chunk_size=4, overlap=1)
            # add chunks to chunk_strings
            chunk_strings.extend(chunks)
            # add metadata for each chunk
            for i, chunk in enumerate(chunks):
                chunk_metadata.append({
                    "movie_idx": doc['id'],
                    "chunk_idx": i,
                    "total_chunks": len(chunks),
                })

        self.chunk_embeddings = self.model.encode(chunk_strings, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        with open("cache/chunk_embeddings.npy", "wb") as f:
            np.save(f, self.chunk_embeddings)

        with open("cache/chunk_metadata.json", "w") as f:
            json.dump(self.chunk_metadata, f)

        return self.chunk_embeddings
    
    def load_or_create_embeddings(self, documents) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}

        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            with open("cache/chunk_embeddings.npy", "rb") as f:
                self.chunk_embeddings = np.load(f)
            with open("cache/chunk_metadata.json", "r") as f:
                self.chunk_metadata = json.load(f)
            if len(self.chunk_metadata) == len(self.chunk_embeddings):
                return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)


    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("No chunk embeddings loaded. Call `load_or_create_embeddings` first.")
        
        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append({
                "chunk_idx": i,
                "movie_idx": self.chunk_metadata[i]["movie_idx"],
                "score": score,
            })

        movie_index_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]
            if movie_idx not in movie_index_scores:
                movie_index_scores[movie_idx] = []
            movie_index_scores[movie_idx].append(score)
        
        # sort the movie scores by score in descending order
        sorted_movie_scores = sorted(movie_index_scores.items(), key=lambda x: max(x[1]), reverse=True)
        filtered_movies_by_limit = sorted_movie_scores[:limit]

        final_output = []
        for movie in filtered_movies_by_limit:
            final_output.append({
                "id": movie[0],
                "title": self.document_map[movie[0]]["title"],
                "document": self.document_map[movie[0]]["description"][:100],
                "score": round(max(movie[1]), 4),
                "metadata": self.document_map[movie[0]].get("metadata", {}) # mot really sure which metadata is expected here?
            })

        return final_output










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
    
    return chunks

def semantic_chunking(text, max_chunk_size=4, overlap=0):
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        sentences = [text]

    chunks = []
    i = 0
    n_sentences = len(sentences)

    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break

        cleaned_sentences = []
        for chunk_sentence in chunk_sentences:
            cleaned_sentences.append(chunk_sentence.strip())
        if not cleaned_sentences:
            continue
        chunk = " ".join(cleaned_sentences)
        chunks.append(chunk)
        i += max_chunk_size - overlap

    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

    return chunks