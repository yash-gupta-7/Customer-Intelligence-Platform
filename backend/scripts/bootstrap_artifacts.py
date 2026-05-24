"""
bootstrap_artifacts.py — Train ML model and build FAISS index if missing (post-deploy).
Run: python scripts/bootstrap_artifacts.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

from app.config import get_settings
from app.ml.pipeline import run_ml_pipeline
from app.rag.pipeline import build_rag_index


def main() -> int:
    settings = get_settings()
    model_path = settings.model_path
    faiss_file = f"{settings.faiss_index_path}.faiss"

    if not os.path.exists(model_path):
        logger.info("ML model not found — running training pipeline…")
        result = run_ml_pipeline(data_path=None, force_promote=True)
        logger.info(f"Training complete: {result.get('message', 'ok')}")
    else:
        logger.info(f"ML model already present at {model_path}")

    if not os.path.exists(faiss_file):
        complaints = settings.complaints_data_path
        if not os.path.exists(complaints):
            logger.error(f"Complaints data missing at {complaints}")
            return 1
        logger.info("FAISS index not found — building RAG index…")
        result = build_rag_index(complaints)
        logger.info(f"Index build complete: {result}")
    else:
        logger.info(f"FAISS index already present at {faiss_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
