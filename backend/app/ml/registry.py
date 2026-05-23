"""
ml/registry.py — MLflow model registry: log, promote, load.
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from loguru import logger
from app.config import get_settings

settings = get_settings()


def setup_mlflow():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    logger.info(f"MLflow tracking URI: {settings.mlflow_tracking_uri}")


def log_run(
    model,
    metrics: dict,
    params: dict,
    feature_names: list[str],
    run_name: str = "training-run",
) -> str:
    """Log a training run to MLflow. Returns run_id."""
    setup_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_param("feature_names", ",".join(feature_names))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=settings.model_registry_name,
        )
        run_id = run.info.run_id
        logger.info(f"Logged run {run_id} with AUC={metrics.get('auc_roc')}")
        return run_id


def promote_model(run_id: str, auc: float) -> bool:
    """
    Promotion gate: if AUC >= threshold, transition the latest model version
    to 'Production'. Returns True if promoted.
    """
    if auc < settings.promotion_auc_gate:
        logger.warning(
            f"Promotion FAILED: AUC {auc:.4f} < gate {settings.promotion_auc_gate}"
        )
        return False

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    versions = client.get_latest_versions(settings.model_registry_name, stages=["None", "Staging"])
    if not versions:
        logger.warning("No model versions found to promote.")
        return False

    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    client.transition_model_version_stage(
        name=settings.model_registry_name,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info(
        f"Model v{latest.version} promoted to Production (AUC={auc:.4f})"
    )
    return True


def get_production_model_uri() -> str | None:
    """Return the URI for the current Production model."""
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    try:
        versions = client.get_latest_versions(
            settings.model_registry_name, stages=["Production"]
        )
        if versions:
            v = versions[0]
            return f"models:/{settings.model_registry_name}/Production"
    except Exception as e:
        logger.warning(f"Could not fetch production model URI: {e}")
    return None
