"""
config.py — Application settings loaded from environment variables.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: str = "development"
    app_name: str = "customer-intelligence-platform"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # ── Security ──────────────────────────────────────────────────────────────
    api_secret_key: str = "dev-secret-key"
    allowed_origins: str = "http://localhost:3000,http://localhost:8080"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "customer-conversion"
    model_registry_name: str = "conversion-model"
    model_stage: str = "Production"

    # ── ML Model ──────────────────────────────────────────────────────────────
    model_path: str = "models/conversion_model.pkl"
    features_path: str = "data/processed/features.parquet"
    reference_data_path: str = "data/processed/reference.parquet"
    promotion_auc_gate: float = 0.75

    # ── RAG ───────────────────────────────────────────────────────────────────
    complaints_data_path: str = "data/complaints/complaints.csv"
    faiss_index_path: str = "models/faiss_index"
    embeddings_model: str = "all-MiniLM-L6-v2"
    rag_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── LLM ───────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    llm_provider: str = "local"
    llm_model: str = "gpt-3.5-turbo"

    # ── Monitoring ────────────────────────────────────────────────────────────
    drift_report_path: str = "reports/drift"


@lru_cache
def get_settings() -> Settings:
    return Settings()
