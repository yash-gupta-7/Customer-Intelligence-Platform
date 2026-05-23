"""
ml/drift.py — Data drift detection using EvidentlyAI.
"""
import os
import json
import pandas as pd
from datetime import datetime
from loguru import logger

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import DatasetDriftMetric
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("EvidentlyAI not installed — drift detection will return stub results.")

from app.config import get_settings

settings = get_settings()


class DriftDetector:
    """
    Compares current production data distribution against a reference dataset.
    Generates an HTML report and returns a structured drift summary.
    """

    def __init__(
        self,
        reference_path: str = None,
        report_dir: str = None,
    ):
        self.reference_path = reference_path or settings.reference_data_path
        self.report_dir = report_dir or settings.drift_report_path
        os.makedirs(self.report_dir, exist_ok=True)

    def _load_reference(self) -> pd.DataFrame | None:
        if not os.path.exists(self.reference_path):
            logger.warning(f"Reference data not found at {self.reference_path}")
            return None
        if self.reference_path.endswith(".parquet"):
            return pd.read_parquet(self.reference_path)
        return pd.read_csv(self.reference_path)

    def detect(self, current_df: pd.DataFrame) -> dict:
        """
        Run drift detection against reference data.
        Returns a dict with: drift_detected, drift_score, n_drifted_features, report_path.
        """
        reference_df = self._load_reference()
        if reference_df is None:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "n_drifted_features": 0,
                "report_path": "",
                "warning": "No reference data found — skipping drift check.",
            }

        if not EVIDENTLY_AVAILABLE:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "n_drifted_features": 0,
                "report_path": "",
                "warning": "EvidentlyAI not installed.",
            }

        # Align columns
        common_cols = [c for c in reference_df.columns if c in current_df.columns]
        ref = reference_df[common_cols]
        cur = current_df[common_cols]

        report = Report(metrics=[DataDriftPreset(), DatasetDriftMetric()])
        report.run(reference_data=ref, current_data=cur)

        # Save HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.report_dir, f"drift_{timestamp}.html")
        report.save_html(report_path)

        # Extract metrics
        result = report.as_dict()
        dataset_drift = result["metrics"][1]["result"]
        drift_detected = dataset_drift.get("dataset_drift", False)
        drift_score = round(dataset_drift.get("share_of_drifted_columns", 0.0), 4)
        n_drifted = dataset_drift.get("number_of_drifted_columns", 0)

        logger.info(
            f"Drift check: detected={drift_detected}, score={drift_score}, "
            f"n_drifted={n_drifted}, report={report_path}"
        )

        return {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "n_drifted_features": n_drifted,
            "report_path": report_path,
        }

    def save_as_reference(self, df: pd.DataFrame):
        """Persist current data as the new reference baseline."""
        os.makedirs(os.path.dirname(self.reference_path), exist_ok=True)
        df.to_parquet(self.reference_path, index=False)
        logger.info(f"Reference data updated: {self.reference_path} ({len(df)} rows)")


_drift_detector: DriftDetector | None = None


def get_drift_detector() -> DriftDetector:
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetector()
    return _drift_detector
