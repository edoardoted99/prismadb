import logging
from .client import get_client

logger = logging.getLogger(__name__)


def get_document_index_name(dataset_id):
    return f"prisma_documents_{dataset_id}"


def get_feature_index_name(run_id):
    return f"prisma_features_{run_id}"


# Known embedding dimensions for HuggingFace models
EMBEDDING_DIMS = {
    'medbit': 768,
    'bio_bert_sentence': 768,
    'sapbert': 768,
    'sbert_multi_minilm': 384,
    'sbert_multi_distiluse': 512,
    'gte_base': 768,
    'gte_large': 1024,
    'gte_multilingual': 768,
    'sbert_mpnet': 768,
    'sbert_minilm': 384,
}


def get_embedding_dim(model_name):
    """
    Return the embedding dimension for a given model name.
    For known HF models, returns a hardcoded value.
    For Ollama/unknown models, probes the model and caches the result.
    Returns None if detection fails.
    """
    if model_name in EMBEDDING_DIMS:
        return EMBEDDING_DIMS[model_name]

    # Probe the model to detect dimension
    try:
        from embeddings.embedders import detect_embedding_dim
        dim = detect_embedding_dim(model_name)
        if dim:
            EMBEDDING_DIMS[model_name] = dim  # cache for future calls
            return dim
    except Exception as e:
        logger.warning(f"Could not detect embedding dim for {model_name}: {e}")

    return None


def create_document_index(dataset_id, embedding_dim):
    """Create an OpenSearch index for documents in a dataset."""
    client = get_client()
    index_name = get_document_index_name(dataset_id)

    if client.indices.exists(index=index_name):
        logger.info(f"Index {index_name} already exists.")
        return

    body = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                "django_id": {"type": "integer"},
                "external_id": {"type": "keyword"},
                "text": {"type": "text", "analyzer": "standard"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                    }
                },
                "status": {"type": "keyword"},
            }
        }
    }

    client.indices.create(index=index_name, body=body)
    logger.info(f"Created index {index_name} with dim={embedding_dim}")


def delete_document_index(dataset_id):
    """Delete the OpenSearch document index for a dataset."""
    client = get_client()
    index_name = get_document_index_name(dataset_id)

    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
        logger.info(f"Deleted index {index_name}")


def create_feature_index(run_id):
    """Create an OpenSearch index for SAE features in a run."""
    client = get_client()
    index_name = get_feature_index_name(run_id)

    if client.indices.exists(index=index_name):
        logger.info(f"Index {index_name} already exists.")
        return

    body = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                "django_id": {"type": "integer"},
                "feature_index": {"type": "integer"},
                "label": {"type": "text"},
                "description": {"type": "text"},
                "density": {"type": "float"},
                "max_activation": {"type": "float"},
                "mean_activation": {"type": "float"},
                "variance_activation": {"type": "float"},
                "example_docs": {"type": "object", "enabled": False},
                "correlated_features": {"type": "object", "enabled": False},
                "co_occurring_features": {"type": "object", "enabled": False},
                "activation_histogram": {"type": "object", "enabled": False},
            }
        }
    }

    client.indices.create(index=index_name, body=body)
    logger.info(f"Created feature index {index_name}")


def delete_feature_index(run_id):
    """Delete the OpenSearch feature index for a run."""
    client = get_client()
    index_name = get_feature_index_name(run_id)

    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
        logger.info(f"Deleted feature index {index_name}")
