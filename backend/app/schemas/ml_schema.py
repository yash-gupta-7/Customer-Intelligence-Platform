"""
schemas/ml_schema.py — Pydantic models for ML service I/O.
"""
from pydantic import BaseModel, Field
from typing import Optional


class CustomerFeatures(BaseModel):
    """Structured customer feature vector for conversion prediction."""
    age: float = Field(..., ge=18, le=100, description="Customer age in years")
    tenure_months: float = Field(..., ge=0, description="Months as a customer")
    account_balance: float = Field(..., description="Current account balance USD")
    num_products: int = Field(..., ge=1, le=10, description="Number of bank products held")
    has_credit_card: int = Field(..., ge=0, le=1, description="1 if has credit card")
    is_active_member: int = Field(..., ge=0, le=1, description="1 if active member")
    estimated_salary: float = Field(..., ge=0, description="Estimated annual salary USD")
    credit_score: float = Field(..., ge=300, le=850, description="Credit score")
    geography: str = Field(..., description="Country/region e.g. France, Germany, Spain")
    gender: str = Field(..., description="Gender: Male or Female")


class ConversionPrediction(BaseModel):
    conversion_probability: float = Field(..., ge=0.0, le=1.0)
    conversion_band: str  # LOW / MEDIUM / HIGH
    model_version: str
    feature_importance: dict[str, float]
    confidence_interval: tuple[float, float]


class TrainRequest(BaseModel):
    retrain: bool = False
    force_promote: bool = False


class TrainResponse(BaseModel):
    run_id: str
    auc_roc: float
    accuracy: float
    promoted: bool
    message: str


class DriftReport(BaseModel):
    drift_detected: bool
    drift_score: float
    n_drifted_features: int
    report_path: str
