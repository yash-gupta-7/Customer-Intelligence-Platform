"""
ml/model.py — Model wrapper: train, evaluate, predict, explain.
"""
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
    average_precision_score,
)
from sklearn.calibration import CalibratedClassifierCV
from loguru import logger


CONVERSION_BANDS = [
    (0.0,  0.40, "LOW"),
    (0.40, 0.70, "MEDIUM"),
    (0.70, 1.01, "HIGH"),
]


def probability_to_band(prob: float) -> str:
    for lo, hi, label in CONVERSION_BANDS:
        if lo <= prob < hi:
            return label
    return "HIGH"


class ConversionModel:
    """
    Gradient Boosting classifier wrapped with:
    - Isotonic calibration for well-calibrated probabilities
    - Feature importance extraction
    - Confidence interval estimation via bootstrap
    """

    DEFAULT_PARAMS = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "min_samples_split": 20,
        "random_state": 42,
    }

    def __init__(self, model_path: str = "models/conversion_model.pkl"):
        self.model_path = model_path
        self.model: CalibratedClassifierCV | None = None
        self.feature_names: list[str] = []
        self.version: str = "untrained"
        self.metrics: dict = {}
        self._load()

    # ── Private ───────────────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(self.model_path):
            artifact = joblib.load(self.model_path)
            self.model = artifact["model"]
            self.feature_names = artifact.get("feature_names", [])
            self.version = artifact.get("version", "unknown")
            self.metrics = artifact.get("metrics", {})
            logger.info(f"Model loaded: version={self.version} metrics={self.metrics}")

    def _build_base(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(**self.DEFAULT_PARAMS)

    # ── Public ────────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        version: str = "v1",
    ) -> dict:
        """Train with calibration. Returns evaluation metrics dict."""
        if len(np.unique(y)) < 2:
            raise ValueError("Training labels must contain at least two classes.")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        base = self._build_base()
        cv_folds = min(5, int(np.bincount(y_train).min()))
        cv_folds = max(cv_folds, 2)
        calibrated = CalibratedClassifierCV(base, method="isotonic", cv=cv_folds)
        calibrated.fit(X_train, y_train)

        y_prob = calibrated.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "auc_roc": round(roc_auc_score(y_val, y_prob), 4),
            "pr_auc": round(average_precision_score(y_val, y_prob), 4),
            "accuracy": round(accuracy_score(y_val, y_pred), 4),
            "precision": round(precision_score(y_val, y_pred), 4),
            "recall": round(recall_score(y_val, y_pred), 4),
            "f1": round(f1_score(y_val, y_pred), 4),
        }
        logger.info(f"Training metrics: {metrics}")

        self.model = calibrated
        self.feature_names = feature_names
        self.version = version
        self.metrics = metrics
        return metrics

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "version": self.version,
                "metrics": self.metrics,
            },
            self.model_path,
        )
        logger.info(f"Model saved to {self.model_path}")

    def predict(self, X: np.ndarray) -> tuple[float, str, dict, tuple[float, float]]:
        """
        Returns: (probability, band, feature_importance_dict, confidence_interval)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Run the ML pipeline first.")

        prob = float(self.model.predict_proba(X)[0, 1])
        band = probability_to_band(prob)

        # Feature importance from base estimator
        importance: dict[str, float] = {}
        try:
            base = self.model.calibrated_classifiers_[0].estimator
            raw_imp = base.feature_importances_
            importance = {
                name: round(float(imp), 4)
                for name, imp in sorted(
                    zip(self.feature_names, raw_imp), key=lambda x: -x[1]
                )
            }
        except Exception:
            importance = {name: 0.0 for name in self.feature_names}

        # Bootstrap confidence interval (fast approximation)
        n_bootstrap = 50
        bootstrap_probs = []
        rng = np.random.default_rng(42)
        for _ in range(n_bootstrap):
            noise = rng.normal(0, 0.02, size=X.shape)
            p = float(self.model.predict_proba(X + noise)[0, 1])
            bootstrap_probs.append(p)
        ci = (
            round(float(np.percentile(bootstrap_probs, 5)), 4),
            round(float(np.percentile(bootstrap_probs, 95)), 4),
        )

        return prob, band, importance, ci


_model_instance: ConversionModel | None = None


def get_model(model_path: str = "models/conversion_model.pkl") -> ConversionModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = ConversionModel(model_path=model_path)
    return _model_instance


def reload_model(model_path: str = "models/conversion_model.pkl") -> ConversionModel:
    """Reload model from disk after training or promotion."""
    global _model_instance
    _model_instance = ConversionModel(model_path=model_path)
    return _model_instance
