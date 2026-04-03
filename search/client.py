import logging

try:
    import chromadb
except ImportError:
    chromadb = None

logger = logging.getLogger(__name__)

_client = None


def get_client():
    """Singleton ChromaDB persistent client."""
    global _client
    if chromadb is None:
        return None
    if _client is None:
        from django.conf import settings
        persist_dir = str(settings.PRISMADB_HOME / "chromadb_data")
        _client = chromadb.PersistentClient(path=persist_dir)
    return _client


def reset_client():
    """Drop the cached client so the next call picks up new settings."""
    global _client
    _client = None


def is_available():
    """Check if ChromaDB is available."""
    try:
        client = get_client()
        if client is None:
            return False
        client.heartbeat()
        return True
    except Exception:
        return False
