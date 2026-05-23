"""
rag/embedder.py — SentenceTransformer embedding wrapper.
"""
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.config import get_settings

settings = get_settings()

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {settings.embeddings_model}")
        _embedder = SentenceTransformer(settings.embeddings_model)
        logger.info("Embedding model loaded.")
    return _embedder


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of text strings. Returns float32 numpy array (N, D)."""
    model = get_embedder()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns float32 array (1, D)."""
    return embed_texts([query])
