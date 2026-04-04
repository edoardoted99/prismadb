import logging

from .client import get_client

logger = logging.getLogger(__name__)


# Known embedding dimensions for common models
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
    'instructor-xl': 768,
}


def get_embedding_dim(model_name):
    """
    Return the embedding dimension for a given model name.
    For known models, returns a hardcoded value.
    For Ollama/unknown models, probes the model and caches the result.
    """
    if model_name in EMBEDDING_DIMS:
        return EMBEDDING_DIMS[model_name]

    try:
        from embeddings.embedders import detect_embedding_dim
        dim = detect_embedding_dim(model_name)
        if dim:
            EMBEDDING_DIMS[model_name] = dim
            return dim
    except Exception as e:
        logger.warning(f"Could not detect embedding dim for {model_name}: {e}")

    return None


def get_document_collection_name(dataset_id):
    return f"prisma_documents_{dataset_id}"


def get_or_create_document_collection(dataset_id):
    """Get or create a ChromaDB collection for a dataset's documents."""
    client = get_client()
    return client.get_or_create_collection(
        name=get_document_collection_name(dataset_id),
        metadata={"hnsw:space": "cosine"},
    )


def delete_document_collection(dataset_id):
    """Delete the ChromaDB collection for a dataset."""
    client = get_client()
    name = get_document_collection_name(dataset_id)
    try:
        client.delete_collection(name)
        logger.info(f"Deleted collection {name}")
    except Exception:
        pass
