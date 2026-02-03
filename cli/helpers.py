
import json
import string
from nltk.stem import PorterStemmer



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