"""
routers/monitor_router.py — Monitoring & Telemetry export endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from prometheus_client import REGISTRY
from loguru import logger
import os
import glob
import json

from app.config import get_settings, Settings
from app.ml.drift import get_drift_detector

router = APIRouter(prefix="/monitoring", tags=["Monitoring & Telemetry"])


@router.get("/export", summary="Export telemetry metrics")
async def export_telemetry(settings: Settings = Depends(get_settings)):
    """
    Collects and exports all Prometheus metrics, drift history, 
    and RAG retrieval summary in a structured JSON payload.
    """
    try:
        metrics_data = {}
        for metric in REGISTRY.collect():
            samples = []
            for sample in metric.samples:
                # Filter out raw system metrics to keep telemetry relevant
                if sample.name.startswith(("cip_", "process_", "python_")):
                    samples.append({
                        "name": sample.name,
                        "labels": sample.labels,
                        "value": sample.value
                    })
            if samples:
                metrics_data[metric.name] = {
                    "help": metric.documentation,
                    "type": metric.type,
                    "samples": samples
                }

        # Collect Evidently drift HTML reports info
        drift_reports = []
        drift_pattern = os.path.join(settings.drift_report_path, "drift_*.html")
        for filepath in glob.glob(drift_pattern):
            drift_reports.append({
                "filename": os.path.basename(filepath),
                "created_at": os.path.getctime(filepath),
                "path": filepath
            })
        # Sort by creation time descending
        drift_reports.sort(key=lambda x: x["created_at"], reverse=True)

        # Collect promotion gate log summary if exists
        promotion_history = []
        if os.path.exists("logs/promotion.log"):
            try:
                with open("logs/promotion.log", "r") as f:
                    for line in f:
                        if line.strip():
                            promotion_history.append(json.loads(line.strip()))
            except Exception:
                pass

        return {
            "status": "success",
            "metrics": metrics_data,
            "drift_reports_count": len(drift_reports),
            "latest_drift_reports": drift_reports[:5],
            "promotion_attempts_count": len(promotion_history),
            "latest_promotion_runs": promotion_history[-5:]
        }
    except Exception as e:
        logger.exception(f"Telemetry export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
