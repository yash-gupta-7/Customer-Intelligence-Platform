"""
schemas/rag_schema.py — Pydantic models for RAG service I/O.
"""
from pydantic import BaseModel, Field
from typing import Optional


class ComplaintQuery(BaseModel):
    """Input for the complaint intelligence RAG service."""
    question: str = Field(..., min_length=5, description="Natural language question about complaints")
    product: Optional[str] = Field(None, description="Filter by product category")
    issue: Optional[str] = Field(None, description="Filter by issue type")
    date_filter: Optional[str] = Field(None, description="ISO date string — filter complaints after this date")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of evidence records to retrieve")


class EvidenceRecord(BaseModel):
    record_id: str
    complaint_text: str
    product: str
    issue: str
    date_received: str
    similarity_score: float
    chunk_index: int


class RAGResponse(BaseModel):
    answer: str
    complaint_themes: list[str]
    cited_record_ids: list[str]
    evidence: list[EvidenceRecord]
    retrieval_count: int
    confidence_score: float


class IndexBuildResponse(BaseModel):
    status: str
    documents_indexed: int
    index_path: str
    message: str
