"""
tests/test_ml.py — Unit tests for ML feature engineering, model, and pipeline.
"""
import numpy as np
import pandas as pd
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ml.features import FeatureEngineer
from app.ml.model import ConversionModel, probability_to_band
from app.ml.pipeline import (
    _generate_synthetic_data,
    stage_validate,
    stage_features,
    stage_train,
    stage_evaluate,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return _generate_synthetic_data(200)


@pytest.fixture
def sample_customer():
    return {
        "age": 35.0,
        "tenure_months": 36.0,
        "account_balance": 75000.0,
        "num_products": 2,
        "has_credit_card": 1,
        "is_active_member": 1,
        "estimated_salary": 65000.0,
        "credit_score": 700.0,
        "geography": "France",
        "gender": "Male",
    }


@pytest.fixture
def trained_model_and_features(sample_df, tmp_path):
    """Train a model on synthetic data using tmp_path for artifacts."""
    model_path = str(tmp_path / "model.pkl")
    scaler_path = str(tmp_path / "scaler.pkl")

    df = stage_validate(sample_df)
    fe = FeatureEngineer(scaler_path=scaler_path)
    X, y, feature_names = (
        lambda df=df: (
            lambda fe=fe, Xy=fe.fit_transform_df(df.drop(columns=["converted"])):
            (*Xy, df["converted"].values)
        )()
    )()

    # Simpler approach
    fe2 = FeatureEngineer(scaler_path=scaler_path)
    X2, feature_names2 = fe2.fit_transform_df(sample_df.drop(columns=["converted"]))
    y2 = sample_df["converted"].values

    model = ConversionModel(model_path=model_path)
    metrics = model.train(X2, y2, feature_names2, version="test-v1")
    return model, X2, feature_names2, metrics


# ── Feature Engineering Tests ─────────────────────────────────────────────────

class TestFeatureEngineer:

    def test_transform_returns_array(self, sample_customer, tmp_path):
        fe = FeatureEngineer(scaler_path=str(tmp_path / "scaler.pkl"))
        df = _generate_synthetic_data(100)
        fe.fit(df.drop(columns=["converted"]))
        X, names = fe.transform(sample_customer)
        assert X.ndim == 2
        assert X.shape[0] == 1
        assert len(names) > 0

    def test_derived_features_present(self, sample_customer, tmp_path):
        fe = FeatureEngineer(scaler_path=str(tmp_path / "scaler.pkl"))
        df = _generate_synthetic_data(100)
        fe.fit(df.drop(columns=["converted"]))
        _, names = fe.transform(sample_customer)
        assert "balance_salary_ratio" in names or "wealth_score" in names

    def test_geography_encoding(self, tmp_path):
        fe = FeatureEngineer(scaler_path=str(tmp_path / "scaler.pkl"))
        df = _generate_synthetic_data(100)
        fe.fit(df.drop(columns=["converted"]))
        for geo in ["France", "Germany", "Spain"]:
            customer = {
                "age": 30.0, "tenure_months": 12.0, "account_balance": 50000.0,
                "num_products": 1, "has_credit_card": 1, "is_active_member": 1,
                "estimated_salary": 50000.0, "credit_score": 650.0,
                "geography": geo, "gender": "Male",
            }
            X, _ = fe.transform(customer)
            assert not np.any(np.isnan(X))


# ── Model Tests ───────────────────────────────────────────────────────────────

class TestConversionModel:

    def test_probability_band_low(self):
        assert probability_to_band(0.2) == "LOW"

    def test_probability_band_medium(self):
        assert probability_to_band(0.55) == "MEDIUM"

    def test_probability_band_high(self):
        assert probability_to_band(0.85) == "HIGH"

    def test_train_returns_metrics(self, trained_model_and_features):
        _, _, _, metrics = trained_model_and_features
        assert "auc_roc" in metrics
        assert "accuracy" in metrics
        assert 0.0 <= metrics["auc_roc"] <= 1.0
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_auc_above_random(self, trained_model_and_features):
        _, _, _, metrics = trained_model_and_features
        assert metrics["auc_roc"] > 0.5, "Model should beat random guessing"

    def test_predict_returns_valid_output(self, trained_model_and_features, tmp_path):
        model, X, _, _ = trained_model_and_features
        prob, band, importance, ci = model.predict(X[:1])
        assert 0.0 <= prob <= 1.0
        assert band in ("LOW", "MEDIUM", "HIGH")
        assert len(importance) > 0
        assert ci[0] <= ci[1]

    def test_model_save_load(self, trained_model_and_features, tmp_path):
        model, X, feature_names, _ = trained_model_and_features
        model.save()
        loaded = ConversionModel(model_path=model.model_path)
        assert loaded.model is not None
        assert loaded.feature_names == feature_names


# ── Pipeline Tests ────────────────────────────────────────────────────────────

class TestMLPipeline:

    def test_validate_drops_nulls(self, sample_df):
        df_with_null = sample_df.copy()
        df_with_null.loc[0, "age"] = None
        validated = stage_validate(df_with_null)
        assert validated["age"].isnull().sum() == 0

    def test_validate_removes_invalid_age(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "age"] = 150  # invalid
        validated = stage_validate(df)
        assert (validated["age"] <= 100).all()

    def test_feature_matrix_shape(self, sample_df):
        df = stage_validate(sample_df)
        X, y, names = stage_features(df)
        assert X.shape[0] == len(df)
        assert X.shape[1] == len(names)
        assert len(y) == len(df)

    def test_stage_gate_relative_promotion(self, tmp_path, monkeypatch):
        # Import target classes and functions
        from app.ml.pipeline import stage_gate
        from app.ml.model import ConversionModel
        from app.config import Settings
        
        # Setup settings overrides for tmp_path
        model_path = str(tmp_path / "conversion_model.pkl")
        
        # 1. Create a dummy baseline model and metrics
        baseline_metrics = {
            "auc_roc": 0.80,
            "pr_auc": 0.70,
            "accuracy": 0.80,
            "precision": 0.80,
            "recall": 0.80,
            "f1": 0.80
        }
        
        # Let's save a baseline model
        baseline_model = ConversionModel(model_path=model_path)
        # Mock actual sklearn model object
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        base = GradientBoostingClassifier()
        calibrated = CalibratedClassifierCV(base, cv=2)
        import numpy as np
        calibrated.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
        baseline_model.model = calibrated
        baseline_model.version = "baseline-v1"
        baseline_model.metrics = baseline_metrics
        baseline_model.save()
        
        # Patch settings path to local model_path
        from app.ml import pipeline
        monkeypatch.setattr(pipeline.settings, "model_path", model_path)
        
        # Case A: Improved model PR-AUC is +1% higher (blocked, requires +3%)
        improved_metrics_low = {
            "auc_roc": 0.81,
            "pr_auc": 0.71, # only +1% improvement
            "f1": 0.81
        }
        
        # Set up a new test model
        new_model = ConversionModel(model_path=model_path)
        new_model.version = "improved-v2"
        
        promoted = stage_gate(new_model, improved_metrics_low, run_id="run123", force=False)
        assert promoted is False, "Should block if PR-AUC improvement is below +3%"
        
        # Case B: Improved model PR-AUC is +5% higher and F1 degrades by 1% (promoted!)
        improved_metrics_high = {
            "auc_roc": 0.84,
            "pr_auc": 0.75, # +5% improvement
            "f1": 0.79 # only 1% degradation (allowed up to 2%)
        }
        
        # Mock promote_model in registry to return True
        from app.ml import registry
        monkeypatch.setattr(registry, "promote_model", lambda run_id, auc: True)
        
        promoted_high = stage_gate(new_model, improved_metrics_high, run_id="run456", force=False)
        assert promoted_high is True, "Should promote if both PR-AUC and F1 criteria are met"
