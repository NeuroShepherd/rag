
import json
import string
from nltk.stem import PorterStemmer
import os
import pickle
from collections import Counter, defaultdict
import math

BM25_K1 = 1.5
BM25_B = 0.75

def search(index, args, movies, stop_words):
    
    # Normalize query to get tokens
    query_tokens = normalize_text(args.query, stop_words=stop_words)
    
    # Collect matching document IDs
    matching_docs = set()
    for token in query_tokens:
        doc_ids = index.get_documents(token)
        matching_docs.update(doc_ids)
    
    # Print results (up to 5)
    for doc_id in sorted(matching_docs)[:5]:
        movie = index.docmap[doc_id]
        print(f"{movie['id']}. {movie['title']}")


def load_movies(file_path: str) -> dict:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["movies"]


def normalize_text(text: str, stop_words: list[str] | None = None) -> list[str]:
    stemmer = PorterStemmer()

    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    text = text.split()
    if stop_words is not None:
        text = [word for word in text if word not in stop_words]
    text = [stemmer.stem(word) for word in text]
    return text


def load_stop_words(file_path: str) -> list[str]:
    with open(file_path, "r") as f:
        stop_words = [f.strip() for f in f.readlines()]
    return stop_words


class InvertedIndex():
    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}
        self.doc_lengths_path =  "cache/doc_lengths.pkl"

    def __add_document(self, doc_id, text):
        tokens = normalize_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
        self.doc_lengths[doc_id] = len(tokens)

    def get_documents(self, token: str):
        value = self.index.get(token, set())
        return sorted(value)
    
    def get_tf(self, doc_id: int, term: str):
        # Normalize the term first
        normalized_term = normalize_text(term)
        if not normalized_term:
            return 0
        normalized_term = normalized_term[0]
        
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id].get(normalized_term, 0)
    
    def get_idf(self, term: str) -> float:
        normalized_term = normalize_text(term)
        if not normalized_term:
            return 0.0
        normalized_term = normalized_term[0]
        total_docs = len(self.docmap)
        docs_with_term = len(self.index.get(normalized_term, []))

        return math.log((total_docs + 1) / (docs_with_term + 1))
    
    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def get_bm25_idf(self, term: str) -> float:
        normalized_term = normalize_text(term)
        if not normalized_term:
            return 0.0
        normalized_term = normalized_term[0]
        total_docs = len(self.docmap)
        docs_with_term = len(self.index.get(normalized_term, []))
        # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        bm25 = math.log((total_docs - docs_with_term + 0.5) / (docs_with_term + 0.5) + 1)
        return bm25
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        tf_component = (tf * (k1 + 1)) / (tf + k1*length_norm)
        return tf_component
    
    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)
        

    def build(self, file_path: str = "data/movies.json"):
        movies = load_movies(file_path)
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)


    def save(self):
        os.makedirs("cache", exist_ok=True)
        
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)

        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()

