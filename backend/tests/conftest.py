"""
Shared pytest fixtures — reset singleton caches between tests.
"""
import pytest


@pytest.fixture(autouse=True)
def reset_singletons():
    """Clear ML/RAG singletons so tests do not leak state."""
    from app.ml import model as model_module
    from app.rag import retriever as retriever_module
    from app.rag import embedder as embedder_module

    model_module._model_instance = None
    retriever_module._retriever = None
    embedder_module._embedder = None
    yield
    model_module._model_instance = None
    retriever_module._retriever = None
    embedder_module._embedder = None
