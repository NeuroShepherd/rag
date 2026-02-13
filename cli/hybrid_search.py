import os
import json
from helpers import InvertedIndex
from semantic_search import ChunkedSemanticSearch


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

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    


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