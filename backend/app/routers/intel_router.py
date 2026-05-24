"""
routers/intel_router.py — Unified POST /customer-intel endpoint.

Fans out to ML service + RAG service in parallel, merges results.
"""
import asyncio
import time
import uuid
from fastapi import APIRouter, HTTPException, Depends, Request
from loguru import logger

from app.auth import require_api_key
from app.config import get_settings, Settings
from app.rate_limit import limiter
from app.ml.model import get_model
from app.ml.features import get_feature_engineer
from app.ml.drift import get_drift_detector
from app.rag.pipeline import run_rag_query
from app.rag.retriever import get_retriever
from app.schemas.intel_schema import (
    CustomerIntelRequest, CustomerIntelResponse, ConfidenceMetrics,
)
from app.monitoring.metrics import record_prediction, record_retrieval
import pandas as pd

router = APIRouter(tags=["Customer Intelligence"])


@router.post(
    "/customer-intel",
    response_model=CustomerIntelResponse,
    summary="Unified customer intelligence — conversion + complaint analysis",
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("10/minute")
async def customer_intel(
    request: Request,
    payload: CustomerIntelRequest,
    settings: Settings = Depends(get_settings),
):
    """
    Unified endpoint that:
    1. Predicts campaign conversion probability (ML service)
    2. Retrieves grounded complaint intelligence (RAG service)
    3. Detects data drift (monitoring)
    4. Returns merged response with confidence metrics

    Input:
      - customer_features: structured customer data
      - product: optional complaint product filter
      - issue: optional complaint issue filter
      - date_filter: optional ISO date cutoff for complaints

    Output:
      - conversion_probability, conversion_band
      - complaint_themes, cited_record_ids, evidence_records
      - confidence_metrics
    """
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    logger.info(f"[{request_id}] /customer-intel request received")

    # ── ML Prediction ─────────────────────────────────────────────────────────
    try:
        scaler_path = settings.model_path.replace("conversion_model.pkl", "scaler.pkl")
        fe = get_feature_engineer(scaler_path)
        X, feature_names = fe.transform(payload.customer_features.model_dump())
        model = get_model(settings.model_path)
        prob, band, importance, ci = model.predict(X)
        model_version = model.version
        record_prediction(prob, band)
        ml_confidence = float((ci[1] - ci[0]))  # narrower CI = higher confidence
        ml_confidence = round(1.0 - min(ml_confidence, 1.0), 4)
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Run POST /ml/train/sync first.",
        )
    except Exception as e:
        logger.exception(f"[{request_id}] ML prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"ML error: {str(e)}")

    # ── RAG Query ─────────────────────────────────────────────────────────────
    rag_answer = "RAG index not available."
    complaint_themes: list[str] = []
    cited_ids: list[str] = []
    evidence_records = []
    rag_confidence = 0.0
    retrieval_count = 0

    try:
        retriever = get_retriever()
        if retriever.index and retriever.index.ntotal > 0:
            # Build a contextual question from customer features
            question = (
                f"What are the main complaints for customers with "
                f"{payload.customer_features.num_products} products, "
                f"credit score {payload.customer_features.credit_score:.0f}, "
                f"and balance ${payload.customer_features.account_balance:,.0f}?"
            )
            if payload.issue:
                question = f"{payload.issue} issues: {question}"

            rag_response = run_rag_query(
                question=question,
                product=payload.product,
                issue=payload.issue,
                date_filter=payload.date_filter,
                top_k=settings.rag_top_k,
            )
            rag_answer = rag_response.answer
            complaint_themes = rag_response.complaint_themes
            cited_ids = rag_response.cited_record_ids
            evidence_records = rag_response.evidence
            rag_confidence = rag_response.confidence_score
            retrieval_count = rag_response.retrieval_count
            record_retrieval(retrieval_count, rag_confidence)
    except Exception as e:
        logger.warning(f"[{request_id}] RAG failed (non-fatal): {e}")

    # ── Drift check (fast, async-friendly) ───────────────────────────────────
    drift_detected = False
    try:
        import os
        if os.path.exists(settings.features_path):
            # Only run drift on a small sample to keep latency low
            df = pd.read_parquet(settings.features_path).sample(min(200, 100))
            drift_result = get_drift_detector().detect(df)
            drift_detected = drift_result.get("drift_detected", False)
    except Exception:
        pass

    # ── Merge & Respond ───────────────────────────────────────────────────────
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        f"[{request_id}] Done in {elapsed_ms}ms | "
        f"prob={prob:.4f} band={band} themes={complaint_themes[:2]}"
    )

    return CustomerIntelResponse(
        conversion_probability=round(prob, 4),
        conversion_band=band,
        feature_importance=importance,
        complaint_themes=complaint_themes,
        cited_record_ids=cited_ids,
        evidence_records=evidence_records,
        rag_answer=rag_answer,
        confidence_metrics=ConfidenceMetrics(
            ml_confidence=ml_confidence,
            rag_confidence=rag_confidence,
            retrieval_count=retrieval_count,
            model_version=model_version,
            drift_detected=drift_detected,
        ),
        request_id=request_id,
        processing_time_ms=elapsed_ms,
    )
