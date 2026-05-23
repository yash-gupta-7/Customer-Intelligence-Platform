"""
ml/features.py — Feature engineering pipeline.

Stages: raw dict → validated DataFrame → encoded → scaled → ready for model.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from loguru import logger


FEATURE_COLUMNS = [
    "credit_score", "age", "tenure_months", "account_balance",
    "num_products", "has_credit_card", "is_active_member",
    "estimated_salary", "geography_encoded", "gender_encoded",
]

GEO_MAP = {"France": 0, "Germany": 1, "Spain": 2}
GENDER_MAP = {"Male": 0, "Female": 1}


class FeatureEngineer:
    """Transforms raw customer dict into a scaled numpy feature vector."""

    def __init__(self, scaler_path: str = "models/scaler.pkl"):
        self.scaler_path = scaler_path
        self.scaler: StandardScaler | None = None
        self._load_scaler()

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_scaler(self):
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"Scaler loaded from {self.scaler_path}")

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["geography_encoded"] = df["geography"].map(GEO_MAP).fillna(-1)
        df["gender_encoded"] = df["gender"].map(GENDER_MAP).fillna(-1)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features."""
        df = df.copy()
        df["balance_salary_ratio"] = df["account_balance"] / (df["estimated_salary"] + 1)
        df["products_per_year"] = df["num_products"] / (df["tenure_months"] / 12 + 1)
        df["wealth_score"] = (
            df["account_balance"] * 0.4
            + df["estimated_salary"] * 0.4
            + df["credit_score"] * 0.2
        )
        return df

    # ── Public ────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Fit scaler on training data. Saves scaler to disk."""
        df = self._encode_categoricals(df)
        df = self._engineer_features(df)
        all_cols = FEATURE_COLUMNS + ["balance_salary_ratio", "products_per_year", "wealth_score"]
        available = [c for c in all_cols if c in df.columns]

        self.scaler = StandardScaler()
        self.scaler.fit(df[available])
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Scaler fitted and saved to {self.scaler_path}")
        return self

    def transform(self, customer: dict) -> tuple[np.ndarray, list[str]]:
        """Transform a single customer dict into a model-ready feature array."""
        df = pd.DataFrame([customer])
        df = self._encode_categoricals(df)
        df = self._engineer_features(df)

        all_cols = FEATURE_COLUMNS + ["balance_salary_ratio", "products_per_year", "wealth_score"]
        available = [c for c in all_cols if c in df.columns]

        X = df[available].values
        if self.scaler:
            X = self.scaler.transform(X)

        return X, available

    def fit_transform_df(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Fit + transform on a full training DataFrame."""
        df = self._encode_categoricals(df)
        df = self._engineer_features(df)

        all_cols = FEATURE_COLUMNS + ["balance_salary_ratio", "products_per_year", "wealth_score"]
        available = [c for c in all_cols if c in df.columns]

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(df[available])
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)

        return X, available


def get_feature_engineer(scaler_path: str = "models/scaler.pkl") -> FeatureEngineer:
    return FeatureEngineer(scaler_path=scaler_path)
