
import json
import string
from nltk.stem import PorterStemmer
import os
import pickle



def search(args, movies, stop_words):
    counter = 1
    print(f"Searching for: {args.query}")
    # breakpoint()
    for movie in (movies):
        # breakpoint()
        query_words = normalize_text(args.query, stop_words=stop_words)
        title_words = normalize_text(movie["title"], stop_words=stop_words)
        if any(q in t for q in query_words for t in title_words):
            print(f"{movie['id']}. Movie Title {movie['title']}")
            counter += 1
            if counter > 5:
                break


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

    def load(self):
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)


def build_command(token: str) -> None:
    index = InvertedIndex()
    index.build()
    index.save()
    docs = index.get_documents(token)
    print(f"First document for token {token} = {docs[0]}")
