"""
rag/retriever.py — FAISS-based vector retriever with metadata filtering.
"""
import os
import json
import pickle
import numpy as np
import faiss
from loguru import logger
from datetime import datetime

from app.config import get_settings
from app.rag.embedder import embed_query

settings = get_settings()


class FAISSRetriever:
    """
    FAISS flat-IP index (inner-product on normalised vectors == cosine similarity).
    Stores document chunks + metadata as a parallel list.
    """

    def __init__(self, index_path: str = None):
        self.index_path = index_path or settings.faiss_index_path
        self.index: faiss.IndexFlatIP | None = None
        self.documents: list[dict] = []   # [{text, record_id, product, issue, date, chunk_idx}]
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        idx_file = self.index_path + ".faiss"
        doc_file = self.index_path + ".pkl"
        if os.path.exists(idx_file) and os.path.exists(doc_file):
            self.index = faiss.read_index(idx_file)
            with open(doc_file, "rb") as f:
                self.documents = pickle.load(f)
            logger.info(
                f"FAISS index loaded: {self.index.ntotal} vectors, "
                f"{len(self.documents)} documents from {self.index_path}"
            )
        else:
            logger.info("No existing FAISS index found — will build on first index call.")

    def save(self):
        os.makedirs(os.path.dirname(self.index_path) if os.path.dirname(self.index_path) else ".", exist_ok=True)
        faiss.write_index(self.index, self.index_path + ".faiss")
        with open(self.index_path + ".pkl", "wb") as f:
            pickle.dump(self.documents, f)
        logger.info(f"FAISS index saved: {self.index.ntotal} vectors → {self.index_path}")

    # ── Indexing ──────────────────────────────────────────────────────────────

    def build(self, documents: list[dict], embeddings: np.ndarray):
        """
        Build FAISS index from list of document dicts and their embeddings.
        documents must contain keys: text, record_id, product, issue, date_received, chunk_index
        """
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.documents = documents
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors (dim={dim})")
        self.save()

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        product_filter: str | None = None,
        issue_filter: str | None = None,
        date_filter: str | None = None,
    ) -> list[dict]:
        """
        Retrieve top-k most similar chunks, with optional metadata filters.
        Returns list of dicts with original metadata + similarity_score.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty — returning no results.")
            return []

        k = top_k or settings.rag_top_k
        # Over-fetch to allow filtering
        fetch_k = min(k * 10, self.index.ntotal)

        q_emb = embed_query(query)
        scores, indices = self.index.search(q_emb, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]

            # Metadata filtering
            if product_filter and product_filter.lower() not in doc.get("product", "").lower():
                continue
            if issue_filter and issue_filter.lower() not in doc.get("issue", "").lower():
                continue
            if date_filter:
                try:
                    doc_date = datetime.fromisoformat(doc.get("date_received", "2000-01-01"))
                    cutoff = datetime.fromisoformat(date_filter)
                    if doc_date < cutoff:
                        continue
                except Exception:
                    pass

            results.append({**doc, "similarity_score": round(float(score), 4)})
            if len(results) >= k:
                break

        logger.info(
            f"Retrieved {len(results)} chunks for query='{query[:60]}' "
            f"(filters: product={product_filter}, issue={issue_filter})"
        )
        return results


_retriever: FAISSRetriever | None = None


def get_retriever() -> FAISSRetriever:
    global _retriever
    if _retriever is None:
        _retriever = FAISSRetriever()
    return _retriever
