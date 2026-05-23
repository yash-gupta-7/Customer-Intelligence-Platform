"""
routers/ml_router.py — ML service endpoints.
"""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, File, UploadFile
from loguru import logger
from typing import List, Optional

from app.config import get_settings, Settings
from app.ml.pipeline import run_ml_pipeline
from app.ml.model import get_model
from app.ml.features import get_feature_engineer
from app.ml.drift import get_drift_detector
from app.schemas.ml_schema import (
    CustomerFeatures, ConversionPrediction,
    TrainRequest, TrainResponse, DriftReport,
)
from app.monitoring.metrics import record_prediction, PIPELINE_RUNS
import pandas as pd

router = APIRouter(prefix="/ml", tags=["ML Service"])


@router.post("/predict", response_model=ConversionPrediction, summary="Predict conversion probability")
async def predict_conversion(
    features: CustomerFeatures,
    settings: Settings = Depends(get_settings),
):
    """
    Predict campaign conversion probability for a single customer.
    Returns probability, band (LOW/MEDIUM/HIGH), feature importances, and CI.
    """
    try:
        fe = get_feature_engineer(settings.model_path.replace("conversion_model.pkl", "scaler.pkl"))
        X, feature_names = fe.transform(features.model_dump())

        model = get_model(settings.model_path)
        prob, band, importance, ci = model.predict(X)

        record_prediction(prob, band)
        logger.info(f"Prediction: prob={prob:.4f}, band={band}")

        return ConversionPrediction(
            conversion_probability=round(prob, 4),
            conversion_band=band,
            model_version=model.version,
            feature_importance=importance,
            confidence_interval=ci,
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready: {str(e)}. Run POST /ml/train first.",
        )
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", response_model=TrainResponse, summary="Trigger ML training pipeline")
async def train_model(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
):
    """
    Trigger the 7-stage ML training pipeline asynchronously.
    Runs: Ingest → Validate → Features → Train → Evaluate → Gate → Serve.
    """
    def _run():
        try:
            result = run_ml_pipeline(
                data_path=settings.features_path,
                force_promote=request.force_promote,
            )
            PIPELINE_RUNS.labels(status="success").inc()
            return result
        except Exception as e:
            PIPELINE_RUNS.labels(status="failure").inc()
            logger.exception(f"Background training failed: {e}")

    background_tasks.add_task(_run)
    return TrainResponse(
        run_id="pending",
        auc_roc=0.0,
        accuracy=0.0,
        promoted=False,
        message="Training pipeline started in background. Check MLflow for results.",
    )


@router.post("/train/sync", response_model=TrainResponse, summary="Train model synchronously")
async def train_model_sync(
    request: TrainRequest,
    settings: Settings = Depends(get_settings),
):
    """Run the ML pipeline synchronously and return metrics immediately."""
    try:
        result = run_ml_pipeline(
            data_path=settings.features_path,
            force_promote=request.force_promote,
        )
        PIPELINE_RUNS.labels(status="success").inc()
        metrics = result.get("metrics", {})
        return TrainResponse(
            run_id=result["run_id"],
            auc_roc=metrics.get("auc_roc", 0.0),
            accuracy=metrics.get("accuracy", 0.0),
            promoted=result["promoted"],
            message=result["message"],
        )
    except Exception as e:
        PIPELINE_RUNS.labels(status="failure").inc()
        logger.exception(f"Sync training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drift", response_model=DriftReport, summary="Run data drift detection")
async def detect_drift(settings: Settings = Depends(get_settings)):
    """Compare current feature distribution against the reference baseline."""
    try:
        if not __import__("os").path.exists(settings.features_path):
            raise HTTPException(
                status_code=404,
                detail="Features data not found. Run /ml/train first.",
            )
        current_df = pd.read_parquet(settings.features_path)
        detector = get_drift_detector()
        report = detector.detect(current_df)
        return DriftReport(**report)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Drift detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info", summary="Get current model metadata")
async def model_info(settings: Settings = Depends(get_settings)):
    """Return loaded model version and feature list."""
    model = get_model(settings.model_path)
    return {
        "version": model.version,
        "feature_names": model.feature_names,
        "model_path": settings.model_path,
        "loaded": model.model is not None,
    }


@router.post("/batch-score", summary="Batch score customer campaign conversions")
async def batch_score(
    features_list: Optional[List[CustomerFeatures]] = None,
    file: Optional[UploadFile] = File(None),
    settings: Settings = Depends(get_settings),
):
    """
    Perform batch campaign conversion predictions.
    Supports either:
      - A JSON array in the request body containing a list of CustomerFeatures objects
      - A CSV file upload containing customer features matching the schema
    """
    records = []
    if features_list is not None:
        records = [item.model_dump() for item in features_list]
        logger.info(f"Received batch score request via JSON with {len(records)} records")
    elif file is not None:
        try:
            import io
            contents = await file.read()
            df = pd.read_csv(io.BytesIO(contents))
            records = df.to_dict(orient="records")
            logger.info(f"Received batch score request via CSV upload with {len(records)} records")
        except Exception as e:
            logger.error(f"Failed to parse CSV upload: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")
    else:
        raise HTTPException(
            status_code=400,
            detail="Missing payload. Must provide either features_list as JSON or a CSV file upload."
        )

    if not records:
        return {
            "status": "success",
            "total_records": 0,
            "predictions": [],
            "message": "No records to process"
        }

    try:
        fe = get_feature_engineer(settings.model_path.replace("conversion_model.pkl", "scaler.pkl"))
        model = get_model(settings.model_path)
        if model.model is None:
            raise HTTPException(
                status_code=503,
                detail="ML model is not loaded. Run POST /ml/train/sync first."
            )

        predictions = []
        for idx, rec in enumerate(records):
            try:
                # Validate and coerce types via Pydantic
                validated = CustomerFeatures(**rec)
                X, _ = fe.transform(validated.model_dump())
                prob, band, importance, ci = model.predict(X)
                
                predictions.append({
                    "record_index": idx,
                    "status": "success",
                    "conversion_probability": round(prob, 4),
                    "conversion_band": band,
                    "confidence_interval": ci
                })
            except Exception as row_err:
                predictions.append({
                    "record_index": idx,
                    "status": "error",
                    "error": str(row_err),
                    "record_raw": rec
                })

        return {
            "status": "success",
            "total_records": len(records),
            "processed_records": len([p for p in predictions if p["status"] == "success"]),
            "model_version": model.version,
            "predictions": predictions
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Batch scoring execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {str(e)}")
