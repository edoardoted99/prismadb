import logging

from .collections import get_or_create_document_collection

logger = logging.getLogger(__name__)


def search_similar_documents(dataset_id, embedding, k=5, exclude_id=None):
    """kNN search for similar documents by embedding vector."""
    collection = get_or_create_document_collection(dataset_id)
    n_results = k + (1 if exclude_id else 0)

    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            if exclude_id and meta["django_id"] == exclude_id:
                continue
            hits.append({
                "django_id": meta["django_id"],
                "external_id": meta.get("external_id", ""),
                "text": results["documents"][0][i] if results.get("documents") is not None else "",
                "score": 1.0 - results["distances"][0][i],  # cosine distance -> similarity
            })
        return hits[:k]
    except Exception as e:
        logger.error(f"kNN search error: {e}")
        return []


def search_documents_bm25(dataset_id, query_text, size=10):
    """Full-text BM25 search on document text."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.error("rank_bm25 not installed. Install with: pip install rank-bm25")
        return []

    collection = get_or_create_document_collection(dataset_id)

    try:
        all_docs = collection.get(include=["documents", "metadatas"])
        if not all_docs["documents"]:
            return []

        # Tokenize and score
        corpus = [doc.lower().split() for doc in all_docs["documents"]]
        bm25 = BM25Okapi(corpus)

        tokenized_query = query_text.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Get top-k by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:size]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "django_id": all_docs["metadatas"][idx]["django_id"],
                    "external_id": all_docs["metadatas"][idx].get("external_id", ""),
                    "text": all_docs["documents"][idx],
                    "score": float(scores[idx]),
                })
        return results
    except Exception as e:
        logger.error(f"BM25 search error: {e}")
        return []


def search_documents_hybrid(dataset_id, query_text, embedding, size=10,
                            bm25_weight=0.3, knn_weight=0.7):
    """Hybrid search combining BM25 full-text and kNN semantic similarity."""
    bm25_results = search_documents_bm25(dataset_id, query_text, size=size * 2)
    knn_results = search_similar_documents(dataset_id, embedding, k=size * 2)

    def _normalize(results):
        if not results:
            return results
        max_score = max(r["score"] for r in results) or 1.0
        for r in results:
            r["norm_score"] = r["score"] / max_score
        return results

    bm25_results = _normalize(bm25_results)
    knn_results = _normalize(knn_results)

    bm25_map = {r["django_id"]: r for r in bm25_results}
    knn_map = {r["django_id"]: r for r in knn_results}

    all_ids = set(bm25_map.keys()) | set(knn_map.keys())
    combined = []
    for did in all_ids:
        bm25_score = bm25_map[did]["norm_score"] if did in bm25_map else 0.0
        knn_score = knn_map[did]["norm_score"] if did in knn_map else 0.0
        entry = bm25_map.get(did) or knn_map.get(did)
        entry["score"] = bm25_weight * bm25_score + knn_weight * knn_score
        combined.append(entry)

    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:size]
