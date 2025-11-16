import os
from collections import defaultdict

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def hybrid_score(self, bm25_score, semantic_score, alpha=0.5):
        return alpha * bm25_score + (1 - alpha) * semantic_score

    def weighted_search(self, query, alpha, limit=5):
        bm25_result = self._bm25_search(query, limit * 500)
        semantic_result = self.semantic_search.search_chunks(query, limit * 500)

        bm25_max = max(item[1] for item in bm25_result)
        bm25_min = min(item[1] for item in bm25_result)
        semantic_max = max(item["score"] for item in semantic_result)
        semantic_min = min(item["score"] for item in semantic_result)

        results = {}

        for doc, score in bm25_result:
            normalized_score = self.normalize_score(score, bm25_min, bm25_max)
            results[doc["id"]] = {
                "document": doc,
                "scores": {"keyword": normalized_score, "semantic": 0.0},
                "hybrid_score": self.hybrid_score(normalized_score, 0.0, alpha),
            }

        for r in semantic_result:
            doc_id = r["id"]
            normalized_score = self.normalize_score(
                r["score"], semantic_min, semantic_max
            )
            if doc_id in results:
                results[doc_id]["scores"]["semantic"] = normalized_score
                results[doc_id]["hybrid_score"] = self.hybrid_score(
                    results[doc_id]["scores"]["keyword"], normalized_score, alpha
                )
            else:
                results[doc_id] = {
                    "document": self.idx.docmap[doc_id],
                    "scores": {"keyword": 0.0, "semantic": normalized_score},
                    "hybrid_score": self.hybrid_score(0.0, normalized_score, alpha),
                }

        sorted_results = sorted(
            results.values(), key=lambda key: key["hybrid_score"], reverse=True
        )
        return sorted_results[:limit]

    def normalize_score(self, score, min_score, max_score):
        return (score - min_score) / (max_score - min_score)

    def rrf_search(self, query, k, limit=10):
        bm25_result = self._bm25_search(query, limit * 500)
        semantic_result = self.semantic_search.search_chunks(query, limit * 500)

        results = {}

        for rank, (doc, _) in enumerate(bm25_result, 1):
            rrf_score = 1 / (k + rank)
            results[doc["id"]] = {
                "document": doc,
                "ranks": {"keyword": rank, "semantic": None},
                "combined_rrf_score": rrf_score,
            }

        for rank, r in enumerate(semantic_result, 1):
            doc_id = r["id"]
            rrf_score = 1 / (k + rank)
            if doc_id in results:
                results[doc_id]["ranks"]["semantic"] = rank
                results[doc_id]["combined_rrf_score"] += rrf_score
            else:
                results[doc_id] = {
                    "document": self.idx.docmap[doc_id],
                    "ranks": {"keyword": None, "semantic": rank},
                    "combined_rrf_score": rrf_score,
                }

        sorted_results = sorted(
            results.values(), key=lambda key: key["combined_rrf_score"], reverse=True
        )
        return sorted_results[:limit]


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0 for _ in range(len(scores))]

    return [(score - min_score) / (max_score - min_score) for score in scores]


def weighed_search(query, alpha, limit):
    hs = HybridSearch(load_movies())
    results = hs.weighted_search(query.strip(), alpha, limit)
    for idx, r in enumerate(results, 1):
        print(f'{idx}. {r["document"]["title"]}')
        print(f'Hybrid Score: {r["hybrid_score"]}')
        print(f'BM25: {r["scores"]["keyword"]}, Semantic: {r["scores"]["semantic"]}')
        print(r["document"]["description"][:200] + "...")
        print()


def rrf_search(query, k, limit):
    hs = HybridSearch(load_movies())
    results = hs.rrf_search(query.strip(), k, limit)
    for idx, r in enumerate(results, 1):
        print(f'{idx}. {r["document"]["title"]}')
        print(f'RRF Score: {r["combined_rrf_score"]:.4f}')
        print(f'BM25: {r["ranks"]["keyword"]}, Semantic: {r["ranks"]["semantic"]}')
        print(r["document"]["description"][:200] + "...")
        print()
