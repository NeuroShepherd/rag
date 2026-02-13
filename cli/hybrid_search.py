import os
import json
from helpers import InvertedIndex
from semantic_search import ChunkedSemanticSearch
from dotenv import load_dotenv
from google import genai


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists("cache/index.pkl"):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25 = self._bm25_search(query, limit*500)
        semantic_results = self.semantic_search.search_chunks(query, limit*500)

        # Combine results using weighted approach
        #  normalize the keyword and semantic scores using normalize_scores function
        bm25_scores = [score for _, score in bm25]
        semantic_scores = [result["score"] for result in semantic_results]
        normalized_bm25 = normalize_scores(*bm25_scores)
        normalized_semantic = normalize_scores(*semantic_scores)

        # Create a mapping of doc_id to normalized scores
        bm25_map = {doc_id: normalized_bm25[i] for i, (doc_id, bm25_score) in enumerate(bm25)}
        semantic_map = {result["id"]: normalized_semantic[i] for i, result in enumerate(semantic_results)}

        # Get all unique doc_ids from both searches
        all_doc_ids = set(bm25_map.keys()) | set(semantic_map.keys())
        
        combined_scores = {}
        for doc_id in all_doc_ids:
            bm25_norm = bm25_map.get(doc_id, 0)
            semantic_norm = semantic_map.get(doc_id, 0)
            combined_scores[doc_id] = {
                "combined": alpha * bm25_norm + (1 - alpha) * semantic_norm,
                "bm25": bm25_norm,
                "semantic": semantic_norm
            }

        # Return top `limit` documents based on combined scores
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1]["combined"], reverse=True)
        return [(doc_id, scores["combined"], scores["bm25"], scores["semantic"]) for doc_id, scores in sorted_docs[:limit]]

    def rrf_search(self, query, k, limit=10, enhance=None):
        if enhance:
            query_original = query
            query = self.enhance_query(query, method=enhance)
            print(f"Enhanced query ({enhance}): '{query_original}' -> '{query}'\n")

        bm25_results = self._bm25_search(query, limit*500)
        semantic_results = self.semantic_search.search_chunks(query, limit*500)
        bm25_map = {doc_id: rank for rank, (doc_id, score) in enumerate(bm25_results)}
        semantic_map = {result["id"]: rank for rank, result in enumerate(semantic_results)}
        all_doc_ids = set(bm25_map.keys()) | set(semantic_map.keys())

        rrf_scores = {}
        for doc_id in all_doc_ids:
            bm25_rank = bm25_map.get(doc_id, float('inf'))
            semantic_rank = semantic_map.get(doc_id, float('inf'))
            rrf_scores[doc_id] = {
                "score": (1 / (k + bm25_rank)) + (1 / (k + semantic_rank)),
                "bm25_rank": bm25_rank,
                "semantic_rank": semantic_rank
            }
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)

        final_output = []
        for doc_id, score_data in sorted_docs[:limit]:
            final_output.append({
                "id": doc_id,
                "title": self.semantic_search.document_map[doc_id]["title"],
                "document": self.semantic_search.document_map[doc_id]["description"][:100],
                "score": round(score_data["score"], 4),
                "metadata": {
                    "bm25_rank": score_data["bm25_rank"] if score_data["bm25_rank"] != float('inf') else "N/A",
                    "semantic_rank": score_data["semantic_rank"] if score_data["semantic_rank"] != float('inf') else "N/A"
                }
            })
        return final_output
    
    def enhance_query(self, query, method):
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        print(f"Using key {api_key[:6]}...")

        model = "gemini-2.5-flash"
        client = genai.Client(api_key=api_key)

        if method == "spell":
            contents = f"""Fix any spelling errors in this movie search query.

                        Only correct obvious typos. Don't change correctly spelled words.

                        Query: "{query}"

                        If no errors, return the original query.
                        Corrected:"""
        
        if method == "rewrite":
            contents = f"""Rewrite this movie search query to be more specific and searchable.

                        Original: "{query}"

                        Consider:
                        - Common movie knowledge (famous actors, popular films)
                        - Genre conventions (horror = scary, animation = cartoon)
                        - Keep it concise (under 10 words)
                        - It should be a google style search query that's very specific
                        - Don't use boolean logic

                        Examples:

                        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                        Rewritten query:"""
            
        if method == "expand":
            contents = f"""Expand this movie search query with related terms.

                        Add synonyms and related concepts that might appear in movie descriptions.
                        Keep expansions relevant and focused.
                        This will be appended to the original query.

                        Examples:

                        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                        - "action movie with bear" -> "action thriller bear chase fight adventure"
                        - "comedy with bear" -> "comedy funny bear humor lighthearted"

                        Query: "{query}"
                        """




        response = client.models.generate_content(
            model=model,
            contents=contents,
        )
        
        return response.text.strip()
    


def normalize_scores(*scores):
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    
    normalized_score = []
    for score in scores:
        normalized_score.append((score - min_score) / (max_score - min_score))
    
    return normalized_score
    
def normalize_scores_text(*scores):
    normalized = normalize_scores(*scores)
    for score in normalized:
        print(f"{score:.4f}")

def weighted_search_text(query, alpha, limit=5):
    with open("data/movies.json", "r") as f:
        data = json.load(f)
    documents = data["movies"]
    
    search = HybridSearch(documents=documents)
    results = search.weighted_search(query, alpha, limit)
    for i, result in enumerate(results):
        title = search.semantic_search.document_map[result[0]]['title']
        print(f"{i+1}. {title}\nHybrid score: {result[1]:.4f}\nBM25: {result[2]:.4f}, Semantic: {result[3]:.4f}\n{search.semantic_search.document_map[result[0]]['description'][:200]}...\n")

def rrf_search_text(query, k, limit=5, enhance=None):
    with open("data/movies.json", "r") as f:
        data = json.load(f)
    documents = data["movies"]
    
    search = HybridSearch(documents=documents)
    results = search.rrf_search(query, k, limit, enhance)
    for i, result in enumerate(results):
        print(f"{i+1}. {result['title']}")
        print(f"RRF score: {result['score']:.4f}")
        print(f"BM25 Rank: {result['metadata'].get('bm25_rank', 'N/A')}, Semantic Rank: {result['metadata'].get('semantic_rank', 'N/A')}")
        print(f"{result['document']}...\n")