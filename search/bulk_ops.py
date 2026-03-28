import logging
from opensearchpy.helpers import bulk, scan
from .client import get_client
from .indices import get_document_index_name, get_feature_index_name

logger = logging.getLogger(__name__)


# =====================================================
# Document Operations
# =====================================================

def bulk_index_documents(dataset_id, documents):
    """
    Index a batch of documents into OpenSearch.
    documents: list of dicts with keys: django_id, external_id, text, status
    """
    client = get_client()
    index_name = get_document_index_name(dataset_id)

    actions = []
    for doc in documents:
        actions.append({
            "_index": index_name,
            "_id": doc['django_id'],
            "_source": {
                "django_id": doc['django_id'],
                "external_id": doc['external_id'],
                "text": doc['text'],
                "status": doc.get('status', 'pending'),
            }
        })

    if actions:
        success, errors = bulk(client, actions, raise_on_error=False)
        if errors:
            logger.error(f"Bulk index errors: {errors[:3]}")
        logger.info(f"Indexed {success} documents into {index_name}")


def update_document_embedding(dataset_id, django_id, embedding):
    """Update a single document's embedding in OpenSearch."""
    client = get_client()
    index_name = get_document_index_name(dataset_id)

    client.update(
        index=index_name,
        id=django_id,
        body={"doc": {"embedding": embedding, "status": "done"}},
    )


def bulk_update_embeddings(dataset_id, updates):
    """
    Bulk update embeddings in OpenSearch.
    updates: list of (django_id, embedding) tuples
    """
    client = get_client()
    index_name = get_document_index_name(dataset_id)

    actions = []
    for django_id, embedding in updates:
        actions.append({
            "_op_type": "update",
            "_index": index_name,
            "_id": django_id,
            "doc": {"embedding": embedding, "status": "done"},
        })

    if actions:
        success, errors = bulk(client, actions, raise_on_error=False)
        if errors:
            logger.error(f"Bulk update errors: {errors[:3]}")
        logger.info(f"Updated {success} embeddings in {index_name}")


def scroll_all_embeddings(dataset_id):
    """
    Generator that yields (django_id, embedding) for all docs with embeddings.
    Used by sae/trainer.py for training data loading.
    """
    client = get_client()
    index_name = get_document_index_name(dataset_id)

    query = {
        "query": {"exists": {"field": "embedding"}},
        "_source": ["django_id", "embedding"],
    }

    for hit in scan(client, index=index_name, query=query, scroll="5m", size=500):
        source = hit["_source"]
        yield source["django_id"], source["embedding"]


def scroll_documents_in_batches(dataset_id, batch_size=512, fields=None):
    """
    Yield batches of document dicts from OpenSearch.
    Each batch is a list of dicts with the requested fields.
    Used by interpreter.py and statistics.py for batch processing.
    """
    client = get_client()
    index_name = get_document_index_name(dataset_id)

    if fields is None:
        fields = ["django_id", "text", "embedding"]

    query = {
        "query": {"exists": {"field": "embedding"}},
        "_source": fields,
    }

    batch = []
    for hit in scan(client, index=index_name, query=query, scroll="5m", size=batch_size):
        batch.append(hit["_source"])
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def count_documents(dataset_id, with_embedding=True):
    """Count documents in an OpenSearch index."""
    client = get_client()
    index_name = get_document_index_name(dataset_id)

    if with_embedding:
        query = {"query": {"exists": {"field": "embedding"}}}
    else:
        query = {"query": {"match_all": {}}}

    result = client.count(index=index_name, body=query)
    return result["count"]


def get_document_embedding(dataset_id, django_id):
    """Get a single document's embedding from OpenSearch."""
    client = get_client()
    index_name = get_document_index_name(dataset_id)

    try:
        result = client.get(index=index_name, id=django_id, _source=["embedding"])
        return result["_source"].get("embedding")
    except Exception:
        return None


def get_document_by_id(dataset_id, django_id):
    """Get full document data from OpenSearch."""
    client = get_client()
    index_name = get_document_index_name(dataset_id)

    try:
        result = client.get(index=index_name, id=django_id)
        return result["_source"]
    except Exception:
        return None


# =====================================================
# Feature Operations
# =====================================================

def bulk_index_features(run_id, features):
    """
    Index features into OpenSearch.
    features: list of dicts with feature data
    """
    client = get_client()
    index_name = get_feature_index_name(run_id)

    actions = []
    for feat in features:
        doc_id = feat.get('django_id', feat.get('feature_index'))
        actions.append({
            "_index": index_name,
            "_id": doc_id,
            "_source": feat,
        })

    if actions:
        success, errors = bulk(client, actions, raise_on_error=False)
        if errors:
            logger.error(f"Feature bulk index errors: {errors[:3]}")
        logger.info(f"Indexed {success} features into {index_name}")


def update_feature(run_id, feature_id, data):
    """Update a single feature in OpenSearch (upsert)."""
    client = get_client()
    index_name = get_feature_index_name(run_id)

    try:
        client.update(
            index=index_name,
            id=feature_id,
            body={"doc": data, "doc_as_upsert": True},
        )
    except Exception as e:
        logger.error(f"Failed to update feature {feature_id}: {e}")


def bulk_update_features(run_id, updates):
    """
    Bulk update features in OpenSearch.
    updates: list of (feature_id, data_dict) tuples
    """
    client = get_client()
    index_name = get_feature_index_name(run_id)

    actions = []
    for feature_id, data in updates:
        actions.append({
            "_op_type": "update",
            "_index": index_name,
            "_id": feature_id,
            "doc": data,
            "doc_as_upsert": True,
        })

    if actions:
        success, errors = bulk(client, actions, raise_on_error=False)
        if errors:
            logger.error(f"Feature bulk update errors: {errors[:3]}")
        logger.info(f"Updated {success} features in {index_name}")
