# embeddings/services.py
import json
import logging
import threading
import time
from typing import IO

from django.db import close_old_connections, connection

from explorer.task_status import TASK_PROGRESS
from project.constants import DOC_DONE, DOC_PENDING
from search.bulk_ops import bulk_add_documents_with_embeddings

from .embedders import get_embedder
from .models import Dataset, Document

logger = logging.getLogger(__name__)


def ingest_json_and_create_dataset(file_obj: IO[bytes], name: str, description: str, model_name: str) -> Dataset:
    raw = file_obj.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON non valido: {e}")

    if not isinstance(data, list):
        raise ValueError("Il JSON deve essere una lista di oggetti {id, text}.")

    dataset = Dataset.objects.create(name=name, description=description, model_name=model_name)

    docs_to_create = []
    for item in data:
        if not isinstance(item, dict): continue
        ext_id = item.get("id")
        text = item.get("text")
        if ext_id is None or text is None: continue

        docs_to_create.append(
            Document(dataset=dataset, external_id=str(ext_id), text=str(text), status=DOC_PENDING)
        )
    Document.objects.bulk_create(docs_to_create, batch_size=500)

    return dataset


def generate_embeddings_for_dataset(dataset_id: int, batch_size: int = 32, progress_callback=None):
    """
    Generate embeddings via Ollama and store them in ChromaDB.
    Documents in SQLite only track status (pending/done/error).
    """
    close_old_connections()

    tid = threading.get_ident()
    TASK_PROGRESS[tid] = {'progress': 0, 'message': 'Starting...', 'start_time': time.time()}

    logger.info(f"[Embeddings] Starting generation for Dataset #{dataset_id}")

    try:
        dataset = Dataset.objects.get(pk=dataset_id)

        TASK_PROGRESS[tid]['message'] = f"Loading model {dataset.model_name}..."
        embedder = get_embedder(dataset.model_name)

        pending_ids = list(dataset.documents.filter(status=DOC_PENDING).values_list('id', flat=True))
        total = len(pending_ids)
        logger.info(f"[Embeddings] Found {total} pending documents.")

        TASK_PROGRESS[tid]['message'] = f"Processing {total} documents..."

        for i in range(0, total, batch_size):
            batch_ids = pending_ids[i : i + batch_size]

            docs = list(Document.objects.filter(id__in=batch_ids).order_by('id'))
            if not docs:
                continue

            texts = [d.text for d in docs]

            try:
                embeddings = embedder.embed_texts(texts)

                # Update status in SQLite (no embedding stored)
                for doc in docs:
                    doc.status = "done"
                    doc.error_message = ""
                Document.objects.bulk_update(docs, ["status", "error_message"])

                # Store embeddings in ChromaDB
                chroma_docs = [
                    {
                        'django_id': doc.id,
                        'external_id': doc.external_id,
                        'text': doc.text,
                        'embedding': emb,
                    }
                    for doc, emb in zip(docs, embeddings) if emb
                ]
                if chroma_docs:
                    bulk_add_documents_with_embeddings(dataset.id, chroma_docs)

                processed_count = i + len(docs)
                progress_pct = int((processed_count / total) * 100)

                logger.info(f"[Embeddings] Processed {processed_count}/{total}")

                TASK_PROGRESS[tid]['progress'] = progress_pct
                TASK_PROGRESS[tid]['message'] = f"Embedding {processed_count}/{total}"

                if progress_callback:
                    progress_callback(processed_count, total)

            except Exception as e:
                logger.error(f"[Embeddings] Batch Error: {e}")
                for doc in docs:
                    doc.status = "error"
                    doc.error_message = str(e)
                Document.objects.bulk_update(docs, ["status", "error_message"])

    except Exception as e:
        logger.error(f"[Embeddings] Critical Error: {e}")
        TASK_PROGRESS[tid]['message'] = f"Error: {str(e)}"
    finally:
        time.sleep(2)
        if tid in TASK_PROGRESS:
            del TASK_PROGRESS[tid]
        connection.close()


def ingest_huggingface_dataset(
    repo_id: str,
    name: str,
    description: str,
    model_name: str = "instructor-xl",
    text_column: str = "abstract",
    id_column: str = "doi",
    embedding_column: str = "embeddings",
    limit: int | None = None,
    offset: int = 0,
    batch_size: int = 500,
    split: str = "train",
    progress_callback=None,
) -> Dataset:
    """
    Ingest a HuggingFace dataset with pre-computed embeddings.
    Metadata goes to SQLite, embeddings go directly to ChromaDB.
    """
    from datasets import load_dataset

    dataset_obj = Dataset.objects.create(
        name=name, description=description, model_name=model_name,
    )

    hf_ds = load_dataset(repo_id, split=split, streaming=True)

    if offset > 0:
        hf_ds = hf_ds.skip(offset)
    if limit is not None:
        hf_ds = hf_ds.take(limit)

    total_ingested = 0
    docs_batch = []
    chroma_batch = []

    for row in hf_ds:
        text = row.get(text_column)
        ext_id = row.get(id_column)
        embedding = row.get(embedding_column)

        if text is None or ext_id is None:
            continue

        embedding_list = list(embedding) if embedding is not None else None

        doc = Document(
            dataset=dataset_obj,
            external_id=str(ext_id),
            text=str(text),
            status=DOC_DONE if embedding_list else DOC_PENDING,
        )
        docs_batch.append(doc)

        if embedding_list:
            chroma_batch.append({
                'django_id': None,  # set after bulk_create
                'external_id': str(ext_id),
                'text': str(text),
                'embedding': embedding_list,
            })

        if len(docs_batch) >= batch_size:
            Document.objects.bulk_create(docs_batch, batch_size=batch_size)
            if chroma_batch:
                for doc_obj, chroma_doc in zip(docs_batch, chroma_batch):
                    chroma_doc['django_id'] = doc_obj.id
                bulk_add_documents_with_embeddings(dataset_obj.id, chroma_batch)

            total_ingested += len(docs_batch)
            if progress_callback:
                progress_callback(total_ingested)

            docs_batch = []
            chroma_batch = []

    # Flush remaining
    if docs_batch:
        Document.objects.bulk_create(docs_batch, batch_size=batch_size)
        if chroma_batch:
            for doc_obj, chroma_doc in zip(docs_batch, chroma_batch):
                chroma_doc['django_id'] = doc_obj.id
            bulk_add_documents_with_embeddings(dataset_obj.id, chroma_batch)
        total_ingested += len(docs_batch)
        if progress_callback:
            progress_callback(total_ingested)

    return dataset_obj
