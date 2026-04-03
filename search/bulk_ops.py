import logging
import random

from .collections import get_or_create_document_collection

logger = logging.getLogger(__name__)


# =====================================================
# Document Operations
# =====================================================

def bulk_add_documents_with_embeddings(dataset_id, documents):
    """
    Add or update documents with embeddings in ChromaDB.
    documents: list of dicts with keys: django_id, external_id, text, embedding
    Called after embedding generation completes for a batch.
    """
    collection = get_or_create_document_collection(dataset_id)

    ids = []
    embeddings = []
    docs_text = []
    metadatas = []

    for doc in documents:
        emb = doc.get('embedding')
        if not emb:
            continue
        ids.append(str(doc['django_id']))
        embeddings.append(emb)
        docs_text.append(doc.get('text', ''))
        metadatas.append({
            'django_id': doc['django_id'],
            'external_id': str(doc.get('external_id', '')),
            'status': 'done',
        })

    if ids:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=docs_text,
            metadatas=metadatas,
        )
        logger.info(f"Upserted {len(ids)} documents into collection for dataset {dataset_id}")


def scroll_all_embeddings(dataset_id):
    """
    Generator yielding (django_id, embedding) for all docs.
    Used by sae/trainer.py for training data loading.
    """
    collection = get_or_create_document_collection(dataset_id)
    total = collection.count()
    batch_size = 500
    offset = 0

    while offset < total:
        results = collection.get(
            offset=offset,
            limit=batch_size,
            include=["embeddings", "metadatas"],
        )
        if not results["ids"]:
            break
        for i in range(len(results["ids"])):
            django_id = results["metadatas"][i]["django_id"]
            embedding = results["embeddings"][i]
            yield django_id, embedding
        offset += len(results["ids"])


def scroll_documents_in_batches(dataset_id, batch_size=512, fields=None):
    """
    Yield batches of document dicts from ChromaDB.
    Each batch is a list of dicts with the requested fields.
    Used by interpreter.py and statistics.py for batch processing.
    """
    collection = get_or_create_document_collection(dataset_id)
    total = collection.count()

    if fields is None:
        fields = ["django_id", "text", "embedding"]

    include = ["metadatas"]
    if "text" in fields:
        include.append("documents")
    if "embedding" in fields:
        include.append("embeddings")

    offset = 0
    while offset < total:
        results = collection.get(
            offset=offset,
            limit=batch_size,
            include=include,
        )
        if not results["ids"]:
            break

        batch = []
        for i in range(len(results["ids"])):
            doc = {"django_id": results["metadatas"][i]["django_id"]}
            if "documents" in include and results.get("documents"):
                doc["text"] = results["documents"][i]
            if "embeddings" in include and results.get("embeddings"):
                doc["embedding"] = results["embeddings"][i]
            # Pass through other metadata
            doc["external_id"] = results["metadatas"][i].get("external_id", "")
            batch.append(doc)

        if batch:
            yield batch
        offset += len(results["ids"])


def count_documents(dataset_id):
    """Count documents in a ChromaDB collection."""
    collection = get_or_create_document_collection(dataset_id)
    return collection.count()


def get_random_documents(dataset_id, k=10):
    """Get k random documents from ChromaDB."""
    collection = get_or_create_document_collection(dataset_id)
    total = collection.count()
    if total == 0:
        return []

    if total <= k:
        results = collection.get(include=["documents", "embeddings", "metadatas"])
        return [
            {
                "django_id": results["metadatas"][i]["django_id"],
                "external_id": results["metadatas"][i].get("external_id", ""),
                "text": results["documents"][i] if results.get("documents") else "",
                "embedding": results["embeddings"][i] if results.get("embeddings") else None,
            }
            for i in range(len(results["ids"]))
        ]

    # Sample random offsets
    indices = random.sample(range(total), min(k, total))
    docs = []
    for idx in indices:
        result = collection.get(
            offset=idx,
            limit=1,
            include=["documents", "embeddings", "metadatas"],
        )
        if result["ids"]:
            docs.append({
                "django_id": result["metadatas"][0]["django_id"],
                "external_id": result["metadatas"][0].get("external_id", ""),
                "text": result["documents"][0] if result.get("documents") else "",
                "embedding": result["embeddings"][0] if result.get("embeddings") else None,
            })
    return docs
