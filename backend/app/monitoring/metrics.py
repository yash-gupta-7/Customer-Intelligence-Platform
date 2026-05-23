"""
monitoring/metrics.py — Prometheus metrics and structured logging setup.
"""
from prometheus_client import Counter, Histogram, Gauge, Summary
from loguru import logger
import sys
import os


# ── Prometheus Metrics ────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "cip_requests_total",
    "Total API requests",
    ["endpoint", "method", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "cip_request_latency_seconds",
    "API request latency",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

ML_PREDICTIONS = Counter(
    "cip_ml_predictions_total",
    "Total ML predictions made",
    ["conversion_band"],
)

ML_PROBABILITY = Histogram(
    "cip_ml_probability",
    "Distribution of conversion probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

RAG_RETRIEVALS = Histogram(
    "cip_rag_retrieval_count",
    "Number of chunks retrieved per RAG query",
    buckets=[1, 2, 3, 5, 7, 10, 15, 20],
)

RAG_CONFIDENCE = Histogram(
    "cip_rag_confidence",
    "RAG answer confidence scores",
    buckets=[0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0],
)

DRIFT_DETECTED = Counter(
    "cip_drift_detected_total",
    "Number of times data drift was detected",
)

PIPELINE_RUNS = Counter(
    "cip_pipeline_runs_total",
    "ML pipeline execution count",
    ["status"],
)

INDEX_SIZE = Gauge(
    "cip_faiss_index_size",
    "Current number of vectors in the FAISS index",
)


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(log_level: str = "INFO"):
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        level=log_level,
        colorize=True,
    )
    logger.add(
        "logs/app.log",
        rotation="100 MB",
        retention="30 days",
        compression="gz",
        level=log_level,
        enqueue=True,
    )
    logger.info(f"Logging configured at level={log_level}")


def record_prediction(probability: float, band: str):
    ML_PREDICTIONS.labels(conversion_band=band).inc()
    ML_PROBABILITY.observe(probability)


def record_retrieval(count: int, confidence: float):
    RAG_RETRIEVALS.observe(count)
    RAG_CONFIDENCE.observe(confidence)
