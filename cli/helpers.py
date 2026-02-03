
import json
import string



def load_movies(file_path: str) -> dict:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["movies"]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    return text