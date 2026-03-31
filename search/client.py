import logging

try:
    from opensearchpy import OpenSearch
except ImportError:
    OpenSearch = None

from django.conf import settings

logger = logging.getLogger(__name__)

_client = None


def get_client():
    """Singleton OpenSearch client, configured from Django settings."""
    global _client
    if OpenSearch is None:
        return None
    if _client is None:
        _client = OpenSearch(
            hosts=[{
                'host': getattr(settings, 'OPENSEARCH_HOST', 'localhost'),
                'port': int(getattr(settings, 'OPENSEARCH_PORT', 9200)),
            }],
            use_ssl=getattr(settings, 'OPENSEARCH_USE_SSL', False),
            verify_certs=False,
            ssl_show_warn=False,
            timeout=30,
        )
    return _client


def is_available():
    """Check if OpenSearch is reachable."""
    try:
        client = get_client()
        if client is None:
            return False
        return client.ping()
    except Exception:
        return False
