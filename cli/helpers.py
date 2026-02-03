
import json
import string
from nltk.stem import PorterStemmer
import os
import pickle



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

    def __add_document(self, doc_id, text):
        tokens = normalize_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, token: str):
        value = self.index.get(token, set())
        return sorted(value)

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


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()
    docs = index.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")
