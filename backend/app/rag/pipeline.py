"""
rag/pipeline.py — Full 8-stage RAG pipeline.

Stages: Load → Clean → Chunk → Embed → Index → Retrieve → Generate → Cite
"""
import os
import re
import uuid
import numpy as np
import pandas as pd
from loguru import logger
from datetime import datetime

from app.config import get_settings
from app.rag.embedder import embed_texts
from app.rag.retriever import get_retriever, FAISSRetriever
from app.rag.generator import generate_answer, extract_themes
from app.schemas.rag_schema import RAGResponse, EvidenceRecord

settings = get_settings()


# ── Synthetic complaints generator ───────────────────────────────────────────

def _generate_synthetic_complaints(n: int = 2000) -> pd.DataFrame:
    """Generate a plausible synthetic CFPB-style complaint dataset."""
    rng = np.random.default_rng(99)

    products = [
        "Credit card", "Mortgage", "Student loan", "Checking account",
        "Auto loan", "Personal loan", "Debt collection", "Money transfer",
    ]
    issues = [
        "Billing disputes", "Incorrect information on report", "Problem with payment",
        "Fraud or scam", "Account access issues", "Closing account",
        "Managing account", "Unauthorized transactions",
    ]
    templates = [
        "I was charged an unexpected fee of ${amount} on my {product} account. "
        "I called customer service and waited {wait} minutes but got no resolution.",
        "My {product} application was denied without explanation. The {issue} "
        "has been ongoing for {days} days and I have received no response.",
        "Unauthorized charges appeared on my {product}. I reported the {issue} "
        "but the company refused to investigate properly.",
        "I have been trying to resolve this {issue} for {days} days. "
        "The {product} department keeps transferring me without helping.",
        "The interest rate on my {product} was changed without notice. "
        "This {issue} is causing significant financial hardship.",
    ]

    records = []
    start = datetime(2020, 1, 1)
    end = datetime(2024, 12, 31)
    date_range = (end - start).days

    for i in range(n):
        product = rng.choice(products)
        issue = rng.choice(issues)
        template = rng.choice(templates)
        text = template.format(
            product=product.lower(),
            issue=issue.lower(),
            amount=int(rng.uniform(10, 500)),
            wait=int(rng.uniform(15, 120)),
            days=int(rng.uniform(3, 90)),
        )
        date = (start + pd.Timedelta(days=int(rng.integers(0, date_range)))).strftime("%Y-%m-%d")
        records.append({
            "complaint_id": f"COMP-{i+1:05d}",
            "product": product,
            "issue": issue,
            "consumer_complaint_narrative": text,
            "date_received": date,
        })

    return pd.DataFrame(records)


# ── Stage implementations ─────────────────────────────────────────────────────

def stage_load(data_path: str | None) -> pd.DataFrame:
    """Stage 1 — Load: read complaints CSV or generate synthetic data."""
    logger.info("▶ RAG Stage 1: Load")
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        logger.info(f"  Loaded {len(df)} complaints from {data_path}")
    else:
        logger.warning("  Complaints CSV not found — generating synthetic data.")
        df = _generate_synthetic_complaints(2000)
        os.makedirs(os.path.dirname(data_path) if data_path else "data/complaints", exist_ok=True)
        save_path = data_path or "data/complaints/complaints.csv"
        df.to_csv(save_path, index=False)
    return df


def stage_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 2 — Clean: drop nulls, normalise text, deduplicate."""
    logger.info("▶ RAG Stage 2: Clean")
    text_col = "consumer_complaint_narrative"
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].str.strip().str.len() > 20]
    df[text_col] = df[text_col].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.drop_duplicates(subset=[text_col])
    logger.info(f"  After cleaning: {len(df)} complaints")
    return df


def stage_chunk(df: pd.DataFrame) -> list[dict]:
    """Stage 3 — Chunk: split long narratives into overlapping chunks."""
    logger.info("▶ RAG Stage 3: Chunk")
    chunks = []
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    text_col = "consumer_complaint_narrative"

    for _, row in df.iterrows():
        text = str(row[text_col])
        record_id = str(row.get("complaint_id", row.name))
        product = str(row.get("product", ""))
        issue = str(row.get("issue", ""))
        date = str(row.get("date_received", ""))

        # Sliding window chunking
        words = text.split()
        if len(words) <= size:
            chunks.append({
                "text": text,
                "record_id": record_id,
                "product": product,
                "issue": issue,
                "date_received": date,
                "chunk_index": 0,
            })
        else:
            start = 0
            chunk_idx = 0
            while start < len(words):
                end = min(start + size, len(words))
                chunk_text = " ".join(words[start:end])
                chunks.append({
                    "text": chunk_text,
                    "record_id": record_id,
                    "product": product,
                    "issue": issue,
                    "date_received": date,
                    "chunk_index": chunk_idx,
                })
                start += size - overlap
                chunk_idx += 1

    logger.info(f"  Created {len(chunks)} chunks from {len(df)} complaints")
    return chunks


def stage_embed(chunks: list[dict]) -> np.ndarray:
    """Stage 4 — Embed: encode chunk texts with SentenceTransformers."""
    logger.info("▶ RAG Stage 4: Embed")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    logger.info(f"  Embedded {len(texts)} chunks → shape {embeddings.shape}")
    return embeddings


def stage_index(chunks: list[dict], embeddings: np.ndarray) -> FAISSRetriever:
    """Stage 5 — Index: build and save FAISS index."""
    logger.info("▶ RAG Stage 5: Index")
    retriever = get_retriever()
    retriever.build(chunks, embeddings)
    logger.info(f"  FAISS index ready with {retriever.index.ntotal} vectors")
    return retriever


def stage_retrieve(
    retriever: FAISSRetriever,
    question: str,
    product: str | None,
    issue: str | None,
    date_filter: str | None,
    top_k: int,
) -> list[dict]:
    """Stage 6 — Retrieve: fetch relevant chunks."""
    logger.info("▶ RAG Stage 6: Retrieve")
    results = retriever.retrieve(
        query=question,
        top_k=top_k,
        product_filter=product,
        issue_filter=issue,
        date_filter=date_filter,
    )
    return results


def stage_generate(question: str, chunks: list[dict]) -> tuple[str, list[str], float]:
    """Stage 7 — Generate: produce answer with themes."""
    logger.info("▶ RAG Stage 7: Generate")
    return generate_answer(question, chunks)


def stage_cite(chunks: list[dict]) -> list[str]:
    """Stage 8 — Cite: collect unique cited record IDs."""
    logger.info("▶ RAG Stage 8: Cite")
    seen = set()
    cited = []
    for c in chunks:
        rid = c.get("record_id", "")
        if rid and rid not in seen:
            cited.append(rid)
            seen.add(rid)
    return cited


# ── Index builder (one-time) ──────────────────────────────────────────────────

def build_rag_index(data_path: str | None = None) -> dict:
    """
    Run stages 1–5 to build the FAISS index from complaint data.
    Call once at startup or when data changes.
    """
    logger.info("═══════ RAG INDEX BUILD START ═══════")
    df = stage_load(data_path or settings.complaints_data_path)
    df = stage_clean(df)
    chunks = stage_chunk(df)
    embeddings = stage_embed(chunks)
    retriever = stage_index(chunks, embeddings)
    logger.info("═══════ RAG INDEX BUILD DONE ═══════")
    return {
        "status": "success",
        "documents_indexed": retriever.index.ntotal,
        "index_path": settings.faiss_index_path,
        "message": f"Indexed {len(chunks)} chunks from {len(df)} complaints.",
    }


# ── Query runner (per-request) ────────────────────────────────────────────────

def run_rag_query(
    question: str,
    product: str | None = None,
    issue: str | None = None,
    date_filter: str | None = None,
    top_k: int | None = None,
) -> RAGResponse:
    """
    Run stages 6–8 (retrieve → generate → cite) for a live query.
    The index must already be built.
    """
    retriever = get_retriever()
    k = top_k or settings.rag_top_k

    chunks = stage_retrieve(retriever, question, product, issue, date_filter, k)
    answer, themes, confidence = stage_generate(question, chunks)

    if confidence == 0.0:
        cited_ids: list[str] = []
        evidence = []
    else:
        cited_ids = stage_cite(chunks)
        evidence = [
            EvidenceRecord(
                record_id=c["record_id"],
                complaint_text=c["text"][:500],
                product=c.get("product", ""),
                issue=c.get("issue", ""),
                date_received=c.get("date_received", ""),
                similarity_score=c.get("similarity_score", 0.0),
                chunk_index=c.get("chunk_index", 0),
            )
            for c in chunks
        ]

    return RAGResponse(
        answer=answer,
        complaint_themes=themes,
        cited_record_ids=cited_ids,
        evidence=evidence,
        retrieval_count=len(chunks),
        confidence_score=confidence,
    )
