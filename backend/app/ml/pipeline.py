"""
ml/pipeline.py — Full 7-stage ML pipeline.

Stages: Ingest → Validate → Features → Train → Evaluate → Gate → Serve
"""
import os
import uuid
import numpy as np
import pandas as pd
from loguru import logger

from app.config import get_settings
from app.ml.features import get_feature_engineer
from app.ml.model import ConversionModel, get_model
from app.ml.registry import log_run, promote_model
from app.ml.drift import get_drift_detector

settings = get_settings()


# ── Synthetic data generator (replaces real CSV in demo mode) ─────────────────

def _generate_synthetic_data(n: int = 5000) -> pd.DataFrame:
    """Generate realistic synthetic bank churn / conversion dataset."""
    rng = np.random.default_rng(42)

    geo = rng.choice(["France", "Germany", "Spain"], n, p=[0.5, 0.25, 0.25])
    gender = rng.choice(["Male", "Female"], n)
    age = rng.integers(18, 75, n)
    tenure = rng.integers(0, 10, n)
    balance = np.where(
        rng.random(n) > 0.3,
        rng.uniform(1000, 250000, n),
        np.zeros(n),
    )
    n_products = rng.integers(1, 5, n)
    has_cc = rng.integers(0, 2, n)
    is_active = rng.integers(0, 2, n)
    salary = rng.uniform(30000, 200000, n)
    credit = rng.integers(350, 850, n)

    # Synthetic target with realistic correlations
    log_odds = (
        -2.0
        + 0.01 * (age - 40)
        + 0.3 * (balance > 100000).astype(float)
        - 0.5 * is_active
        + 0.2 * (geo == "Germany").astype(float)
        - 0.1 * n_products
        + 0.001 * credit
    )
    prob = 1 / (1 + np.exp(-log_odds))
    converted = rng.binomial(1, prob)

    return pd.DataFrame({
        "geography": geo,
        "gender": gender,
        "age": age.astype(float),
        "tenure_months": tenure.astype(float),
        "account_balance": balance,
        "num_products": n_products.astype(int),
        "has_credit_card": has_cc.astype(int),
        "is_active_member": is_active.astype(int),
        "estimated_salary": salary,
        "credit_score": credit.astype(float),
        "converted": converted.astype(int),
    })


# ── Stage implementations ─────────────────────────────────────────────────────

def stage_ingest(data_path: str | None) -> pd.DataFrame:
    """Stage 1 — Ingest: load CSV or generate synthetic data."""
    logger.info("▶ Stage 1: Ingest")
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        logger.info(f"  Loaded {len(df)} rows from {data_path}")
    else:
        logger.warning("  Data file not found — generating synthetic dataset.")
        df = _generate_synthetic_data(5000)
        os.makedirs("data/processed", exist_ok=True)
        df.to_parquet("data/processed/features.parquet", index=False)
    return df


def stage_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 2 — Validate: schema + null + range checks."""
    logger.info("▶ Stage 2: Validate")
    required = [
        "age", "tenure_months", "account_balance", "num_products",
        "has_credit_card", "is_active_member", "estimated_salary",
        "credit_score", "geography", "gender", "converted",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Validation failed — missing columns: {missing}")

    null_counts = df[required].isnull().sum()
    if null_counts.any():
        logger.warning(f"  Nulls detected:\n{null_counts[null_counts > 0]}")
        df = df.dropna(subset=required)

    before = len(df)
    df = df[(df["age"] >= 18) & (df["age"] <= 100)]
    df = df[(df["credit_score"] >= 300) & (df["credit_score"] <= 850)]
    logger.info(f"  Validation passed. Rows before={before}, after={len(df)}")
    return df


def stage_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Stage 3 — Feature Engineering."""
    logger.info("▶ Stage 3: Feature Engineering")
    fe = get_feature_engineer()
    y = df["converted"].values
    X, feature_names = fe.fit_transform_df(df.drop(columns=["converted"]))
    logger.info(f"  Feature matrix: {X.shape}, features={feature_names}")
    return X, y, feature_names


def stage_train(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> tuple[ConversionModel, dict]:
    """Stage 4 — Train."""
    logger.info("▶ Stage 4: Train")
    version = f"v_{uuid.uuid4().hex[:8]}"
    model = ConversionModel(model_path=settings.model_path)
    metrics = model.train(X, y, feature_names, version=version)
    return model, metrics


def stage_evaluate(metrics: dict) -> dict:
    """Stage 5 — Evaluate: log and return metrics."""
    logger.info("▶ Stage 5: Evaluate")
    logger.info(f"  AUC={metrics['auc_roc']}, Accuracy={metrics['accuracy']}, F1={metrics['f1']}")
    return metrics


def stage_gate(model: ConversionModel, metrics: dict, run_id: str, force: bool = False) -> bool:
    """
    Stage 6 — Relative Promotion Gate.
    Compares improved model metrics against baseline model.
    Thresholds:
      - PR-AUC must improve by at least +3% (+0.03)
      - F1-Score degradation must be no worse than -2% (-0.02)
    """
    import os
    import json
    from datetime import datetime

    logger.info("▶ Stage 6: Promotion Gate")
    improved_pr_auc = metrics.get("pr_auc", 0.0)
    improved_f1 = metrics.get("f1", 0.0)
    improved_auc = metrics.get("auc_roc", 0.0)

    # 1. Try loading baseline model from disk
    baseline_metrics = None
    baseline_version = "none"
    if os.path.exists(settings.model_path):
        try:
            from app.ml.model import ConversionModel
            # Load baseline model into temporary instance
            baseline = ConversionModel(model_path=settings.model_path)
            if baseline.model is not None and baseline.metrics:
                baseline_metrics = baseline.metrics
                baseline_version = baseline.version
                logger.info(f"Loaded baseline model v{baseline_version} metrics: {baseline_metrics}")
        except Exception as e:
            logger.warning(f"Failed to load baseline model metrics for relative comparison: {e}")

    status = "PROMOTED"
    reason = "Passed relative promotion gate"
    promoted = False

    # 2. Apply relative promotion logic
    if force:
        logger.warning("  Force promotion enabled — bypassing relative checks.")
        promoted = True
        status = "PROMOTED_FORCED"
        reason = "Forced promotion bypassed checks"
    elif baseline_metrics is not None:
        baseline_pr_auc = baseline_metrics.get("pr_auc", 0.0)
        baseline_f1 = baseline_metrics.get("f1", 0.0)

        pr_auc_diff = improved_pr_auc - baseline_pr_auc
        f1_diff = improved_f1 - baseline_f1

        pr_auc_passed = pr_auc_diff >= 0.03
        f1_passed = f1_diff >= -0.02

        logger.info(f"Relative gate comparison with baseline v{baseline_version}:")
        logger.info(f"  PR-AUC: baseline={baseline_pr_auc:.4f}, improved={improved_pr_auc:.4f} (diff={pr_auc_diff:+.4f}, required >= +0.03)")
        logger.info(f"  F1-Score: baseline={baseline_f1:.4f}, improved={improved_f1:.4f} (diff={f1_diff:+.4f}, required >= -0.02)")

        if not pr_auc_passed:
            promoted = False
            status = "BLOCKED"
            reason = f"PR-AUC improvement ({pr_auc_diff:+.4f}) is below the required +0.03 threshold"
            logger.warning(f"  Promotion BLOCKED: {reason}")
        elif not f1_passed:
            promoted = False
            status = "BLOCKED"
            reason = f"F1-Score degradation ({f1_diff:+.4f}) is worse than the allowed -0.02 drop"
            logger.warning(f"  Promotion BLOCKED: {reason}")
        else:
            promoted = promote_model(run_id, improved_auc)
            if not promoted:
                status = "FAILED"
                reason = "MLflow model promotion failed"
    else:
        # No baseline exists (first model): apply absolute gate
        logger.info(f"  No baseline model found. Applying absolute gate (AUC >= {settings.promotion_auc_gate}).")
        if improved_auc >= settings.promotion_auc_gate:
            promoted = promote_model(run_id, improved_auc)
            if promoted:
                reason = f"First model promoted: absolute gate passed (AUC={improved_auc:.4f})"
            else:
                status = "FAILED"
                reason = "MLflow model promotion failed"
        else:
            promoted = False
            status = "BLOCKED"
            reason = f"First model blocked: absolute gate failed (AUC={improved_auc:.4f} < {settings.promotion_auc_gate})"
            logger.warning(f"  Promotion BLOCKED: {reason}")

    # 3. Log results to logs/promotion.log in JSONL format
    os.makedirs("logs", exist_ok=True)
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "status": status,
        "baseline_version": baseline_version,
        "baseline_metrics": baseline_metrics or {},
        "improved_version": model.version,
        "improved_metrics": metrics,
        "reason": reason,
    }
    try:
        with open("logs/promotion.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        logger.info(f"Promotion gate results logged successfully.")
    except Exception as e:
        logger.warning(f"Failed to write to logs/promotion.log: {e}")

    return promoted


def stage_serve(model: ConversionModel):
    """Stage 7 — Serve: save model artifact to disk."""
    logger.info("▶ Stage 7: Serve")
    model.save()
    logger.info(f"  Model artifact saved to {settings.model_path}")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_ml_pipeline(
    data_path: str | None = None,
    force_promote: bool = False,
) -> dict:
    """
    Execute all 7 ML pipeline stages.
    Returns a result dict with run_id, metrics, promoted flag.
    """
    logger.info("═══════════════ ML PIPELINE START ═══════════════")
    try:
        df = stage_ingest(data_path)
        df = stage_validate(df)
        X, y, feature_names = stage_features(df)
        model, metrics = stage_train(X, y, feature_names)
        metrics = stage_evaluate(metrics)

        # Log to MLflow
        try:
            run_id = log_run(
                model.model,
                metrics,
                params=ConversionModel.DEFAULT_PARAMS,
                feature_names=feature_names,
                run_name="ml-pipeline",
            )
        except Exception as mlf_err:
            logger.warning(f"MLflow logging failed (non-fatal): {mlf_err}")
            run_id = f"local_{uuid.uuid4().hex[:8]}"

        promoted = stage_gate(model, metrics, run_id, force=force_promote)
        stage_serve(model)

        # Save reference data for drift detection
        try:
            drift_df = pd.read_parquet("data/processed/features.parquet")
            get_drift_detector().save_as_reference(drift_df)
        except Exception:
            pass

        result = {
            "run_id": run_id,
            "metrics": metrics,
            "promoted": promoted,
            "model_version": model.version,
            "message": "Pipeline completed successfully.",
        }
        logger.info(f"═══════════════ ML PIPELINE DONE: {result} ═══════════════")
        return result

    except Exception as e:
        logger.exception(f"ML Pipeline failed: {e}")
        raise
