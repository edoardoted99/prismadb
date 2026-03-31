import logging

from .client import get_client
from .indices import get_document_index_name, get_feature_index_name

logger = logging.getLogger(__name__)


def search_similar_documents(dataset_id, embedding, k=5, exclude_id=None):
    """kNN search for similar documents by embedding vector."""
    client = get_client()
    index_name = get_document_index_name(dataset_id)

    knn_clause = {
        "knn": {
            "embedding": {
                "vector": embedding,
                "k": k + (1 if exclude_id else 0),
            }
        }
    }

    if exclude_id:
        query = {
            "size": k + 1,
            "query": {
                "bool": {
                    "must": [knn_clause],
                    "must_not": [{"term": {"django_id": exclude_id}}],
                }
            },
            "_source": ["django_id", "external_id", "text"],
        }
    else:
        query = {
            "size": k,
            "query": knn_clause,
            "_source": ["django_id", "external_id", "text"],
        }

    try:
        result = client.search(index=index_name, body=query)
        hits = []
        for hit in result["hits"]["hits"]:
            hits.append({
                "django_id": hit["_source"]["django_id"],
                "external_id": hit["_source"].get("external_id"),
                "text": hit["_source"].get("text", ""),
                "score": hit["_score"],
            })
        return hits[:k]
    except Exception as e:
        logger.error(f"kNN search error: {e}")
        return []


def search_documents_bm25(dataset_id, query_text, size=10):
    """Full-text BM25 search on document text."""
    client = get_client()
    index_name = get_document_index_name(dataset_id)

    query = {
        "size": size,
        "query": {
            "match": {
                "text": query_text
            }
        },
        "_source": ["django_id", "external_id", "text"],
    }

    try:
        result = client.search(index=index_name, body=query)
        return [
            {
                "django_id": hit["_source"]["django_id"],
                "external_id": hit["_source"].get("external_id"),
                "text": hit["_source"].get("text", ""),
                "score": hit["_score"],
            }
            for hit in result["hits"]["hits"]
        ]
    except Exception as e:
        logger.error(f"BM25 search error: {e}")
        return []


def search_documents_hybrid(dataset_id, query_text, embedding, size=10,
                            bm25_weight=0.3, knn_weight=0.7):
    """Hybrid search combining BM25 full-text and kNN semantic similarity."""
    client = get_client()
    index_name = get_document_index_name(dataset_id)

    query = {
        "size": size,
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "text": {
                                "query": query_text,
                                "boost": bm25_weight,
                            }
                        }
                    },
                    {
                        "knn": {
                            "embedding": {
                                "vector": embedding,
                                "k": size,
                                "boost": knn_weight,
                            }
                        }
                    }
                ]
            }
        },
        "_source": ["django_id", "external_id", "text"],
    }

    try:
        result = client.search(index=index_name, body=query)
        return [
            {
                "django_id": hit["_source"]["django_id"],
                "external_id": hit["_source"].get("external_id"),
                "text": hit["_source"].get("text", ""),
                "score": hit["_score"],
            }
            for hit in result["hits"]["hits"]
        ]
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        return []


def get_random_documents(dataset_id, k=10, must_have_embedding=True):
    """Get random documents using function_score + random_score."""
    client = get_client()
    index_name = get_document_index_name(dataset_id)

    query_body = {"match_all": {}}
    if must_have_embedding:
        query_body = {"exists": {"field": "embedding"}}

    query = {
        "size": k,
        "query": {
            "function_score": {
                "query": query_body,
                "random_score": {},
            }
        },
        "_source": ["django_id", "external_id", "text", "embedding"],
    }

    try:
        result = client.search(index=index_name, body=query)
        return [hit["_source"] for hit in result["hits"]["hits"]]
    except Exception as e:
        logger.error(f"Random docs error: {e}")
        return []


def search_features(run_id, query_text, size=50):
    """Search features by label or description text."""
    client = get_client()
    index_name = get_feature_index_name(run_id)

    query = {
        "size": size,
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": ["label^3", "description"],
                "type": "best_fields",
            }
        },
        "_source": [
            "django_id", "feature_index", "label", "description",
            "density", "max_activation", "mean_activation", "variance_activation",
        ],
    }

    try:
        result = client.search(index=index_name, body=query)
        return [
            {**hit["_source"], "score": hit["_score"]}
            for hit in result["hits"]["hits"]
        ]
    except Exception as e:
        logger.error(f"Feature search error: {e}")
        return []
