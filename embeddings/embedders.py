import logging

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """Embedder that uses Ollama's /api/embed endpoint."""

    _instances = {}

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        logger.info(f"[OllamaEmbedder] Using model: {model_name} at {self.base_url}")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        timeout = getattr(settings, 'EXPLORER_OLLAMA_TIMEOUT', 300)

        # Try batch /api/embed first (Ollama >= 0.1.44)
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": texts},
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings", [])
            if embeddings and len(embeddings) == len(texts):
                return embeddings
        except Exception as e:
            logger.debug(f"[OllamaEmbedder] /api/embed failed ({e}), trying /api/embeddings fallback")

        # Fallback: single /api/embeddings (older Ollama versions)
        all_embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": text},
                    timeout=timeout,
                )
                response.raise_for_status()
                data = response.json()
                all_embeddings.append(data.get("embedding", []))
            except Exception as e:
                logger.error(f"[OllamaEmbedder] Embedding error: {e}")
                all_embeddings.append([])
        return all_embeddings


def get_embedder(model_name: str):
    """Factory: returns an OllamaEmbedder instance (cached)."""
    if model_name not in OllamaEmbedder._instances:
        OllamaEmbedder._instances[model_name] = OllamaEmbedder(model_name)
    return OllamaEmbedder._instances[model_name]


def detect_embedding_dim(model_name: str) -> int | None:
    """Probe the model to detect its embedding dimension."""
    try:
        embedder = get_embedder(model_name)
        result = embedder.embed_texts(["test"])
        if result and result[0]:
            return len(result[0])
    except Exception as e:
        logger.warning(f"[Embedder] Could not detect dim for {model_name}: {e}")
    return None
