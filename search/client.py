import logging

try:
    from opensearchpy import OpenSearch
except ImportError:
    OpenSearch = None

logger = logging.getLogger(__name__)

_client = None


def get_client():
    """Singleton OpenSearch client, configured from DB settings."""
    global _client
    if OpenSearch is None:
        return None
    if _client is None:
        from project.utils import get_setting
        _client = OpenSearch(
            hosts=[{
                'host': get_setting('opensearch_host'),
                'port': int(get_setting('opensearch_port')),
            }],
            use_ssl=get_setting('opensearch_use_ssl').lower() in ('true', '1', 'yes'),
            verify_certs=False,
            ssl_show_warn=False,
            timeout=30,
        )
    return _client


def reset_client():
    """Drop the cached client so the next call picks up new settings."""
    global _client
    _client = None


def is_available():
    """Check if OpenSearch is reachable."""
    try:
        client = get_client()
        if client is None:
            return False
        return client.ping()
    except Exception:
        return False
