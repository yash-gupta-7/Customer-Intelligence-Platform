"""
schemas/intel_schema.py — Unified /customer-intel endpoint schema.
"""
from pydantic import BaseModel, Field
from typing import Optional
from .ml_schema import CustomerFeatures
from .rag_schema import EvidenceRecord


class CustomerIntelRequest(BaseModel):
    """Unified request payload for the /customer-intel endpoint."""
    customer_features: CustomerFeatures
    product: Optional[str] = Field(None, description="Product filter for complaint search")
    issue: Optional[str] = Field(None, description="Issue type filter for complaint search")
    date_filter: Optional[str] = Field(None, description="Complaints after this ISO date")


class ConfidenceMetrics(BaseModel):
    ml_confidence: float
    rag_confidence: float
    retrieval_count: int
    model_version: str
    drift_detected: bool


class CustomerIntelResponse(BaseModel):
    """Unified response from the /customer-intel endpoint."""
    # ML outputs
    conversion_probability: float = Field(..., ge=0.0, le=1.0)
    conversion_band: str  # LOW / MEDIUM / HIGH
    feature_importance: dict[str, float]

    # RAG outputs
    complaint_themes: list[str]
    cited_record_ids: list[str]
    evidence_records: list[EvidenceRecord]
    rag_answer: str

    # Shared
    confidence_metrics: ConfidenceMetrics
    request_id: str
    processing_time_ms: float
