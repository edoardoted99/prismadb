# embeddings/services.py
import json
import time
from typing import IO

from django.db import close_old_connections, connection

from project.constants import DOC_PENDING

from .embedders import get_embedder
from .models import Dataset, Document


def ingest_json_and_create_dataset(file_obj: IO[bytes], name: str, description: str, model_name: str) -> Dataset:
    # ... (questo rimane uguale, non toccarlo) ...
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

    # ChromaDB indexing deferred to generate_embeddings_for_dataset()
    # (documents need embeddings before they can be stored in ChromaDB)

    return dataset


import logging
import threading

from explorer.task_status import TASK_PROGRESS

logger = logging.getLogger(__name__)

def generate_embeddings_for_dataset(dataset_id: int, batch_size: int = 32, progress_callback=None):
    """
    Versione thread-safe e corretta del generatore.
    progress_callback: optional callable(processed, total) for CLI progress bars.
    """
    # Chiudiamo vecchie connessioni per sicurezza nel thread
    close_old_connections()

    tid = threading.get_ident()
    TASK_PROGRESS[tid] = {'progress': 0, 'message': 'Starting...', 'start_time': time.time()}

    logger.info(f"[Embeddings] Starting generation for Dataset #{dataset_id}")

    try:
        # Ricarichiamo il dataset dentro il thread
        dataset = Dataset.objects.get(pk=dataset_id)

        # Update status
        TASK_PROGRESS[tid]['message'] = f"Loading model {dataset.model_name}..."

        # get_embedder now returns an INSTANCE
        embedder = get_embedder(dataset.model_name)

        # 1. Recuperiamo TUTTI gli ID pending (lista statica) per evitare problemi di slicing dinamico
        pending_ids = list(dataset.documents.filter(status=DOC_PENDING).values_list('id', flat=True))
        total = len(pending_ids)
        logger.info(f"[Embeddings] Found {total} pending documents.")

        TASK_PROGRESS[tid]['message'] = f"Processing {total} documents..."

        # 2. Iteriamo a blocchi sulla lista fissa di ID
        for i in range(0, total, batch_size):
            batch_ids = pending_ids[i : i + batch_size]

            # Recuperiamo gli oggetti Document reali
            docs = list(Document.objects.filter(id__in=batch_ids).order_by('id'))
            if not docs:
                continue

            texts = [d.text for d in docs]

            try:
                # Questa chiamata ora ritorna LISTA DI LISTE (chunks) grazie al nuovo embedder
                embeddings = embedder.embed_texts(texts)

                # Salvataggio
                for doc, emb in zip(docs, embeddings):
                    doc.embedding = emb # Django JSONField gestisce liste di liste automaticamente
                    doc.status = "done"
                    doc.error_message = ""

                # Bulk update per velocità
                Document.objects.bulk_update(docs, ["embedding", "status", "error_message"])

                # Store documents with embeddings in ChromaDB
                try:
                    from search.client import is_available
                    if is_available():
                        from search.bulk_ops import bulk_add_documents_with_embeddings
                        chroma_docs = [
                            {
                                'django_id': doc.id,
                                'external_id': doc.external_id,
                                'text': doc.text,
                                'embedding': doc.embedding,
                            }
                            for doc in docs if doc.status == 'done' and doc.embedding
                        ]
                        if chroma_docs:
                            bulk_add_documents_with_embeddings(dataset.id, chroma_docs)
                except Exception as chroma_err:
                    logger.warning(f"[ChromaDB] Failed to index documents: {chroma_err}")

                processed_count = i + len(docs)
                progress_pct = int((processed_count / total) * 100)

                logger.info(f"[Embeddings] Processed {processed_count}/{total}")

                # Update Task Progress
                TASK_PROGRESS[tid]['progress'] = progress_pct
                TASK_PROGRESS[tid]['message'] = f"Embedding {processed_count}/{total}"

                if progress_callback:
                    progress_callback(processed_count, total)

            except Exception as e:
                logger.error(f"[Embeddings] Batch Error: {e}")
                # Segna come errore ma non fermare tutto il processo
                for doc in docs:
                    doc.status = "error"
                    doc.error_message = str(e)
                Document.objects.bulk_update(docs, ["status", "error_message"])

    except Exception as e:
        logger.error(f"[Embeddings] Critical Error: {e}")
        TASK_PROGRESS[tid]['message'] = f"Error: {str(e)}"
    finally:
        # Cleanup task progress after a short delay or immediately
        # Leaving it for a moment allows the UI to show 100%
        time.sleep(2)
        if tid in TASK_PROGRESS:
            del TASK_PROGRESS[tid]
        connection.close()
