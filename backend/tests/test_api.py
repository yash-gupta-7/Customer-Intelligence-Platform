"""
tests/test_api.py — Integration tests for FastAPI endpoints using TestClient.
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ── Health & Root ─────────────────────────────────────────────────────────────

class TestHealthEndpoints:

    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_root_lists_endpoints(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "endpoints" in data
        assert "unified" in data["endpoints"]

    def test_docs_accessible(self):
        response = client.get("/docs")
        assert response.status_code == 200


# ── ML Endpoints ──────────────────────────────────────────────────────────────

SAMPLE_FEATURES = {
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


class TestMLEndpoints:

    def test_train_sync_succeeds(self):
        response = client.post("/ml/train/sync", json={"retrain": False, "force_promote": True})
        assert response.status_code == 200
        data = response.json()
        assert "run_id" in data
        assert "auc_roc" in data
        assert data["auc_roc"] > 0

    def test_predict_returns_probability(self):
        # Ensure model is trained first
        client.post("/ml/train/sync", json={"retrain": False, "force_promote": True})
        response = client.post("/ml/predict", json=SAMPLE_FEATURES)
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["conversion_probability"] <= 1.0
        assert data["conversion_band"] in ("LOW", "MEDIUM", "HIGH")
        assert len(data["feature_importance"]) > 0

    def test_predict_validates_input(self):
        bad_payload = {**SAMPLE_FEATURES, "age": 10}  # age < 18
        response = client.post("/ml/predict", json=bad_payload)
        assert response.status_code == 422

    def test_model_info_returns_version(self):
        client.post("/ml/train/sync", json={"retrain": False, "force_promote": True})
        response = client.get("/ml/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert data["loaded"] is True


# ── RAG Endpoints ─────────────────────────────────────────────────────────────

class TestRAGEndpoints:

    def test_build_index_sync(self):
        response = client.post("/rag/index/build/sync")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["documents_indexed"] > 0

    def test_index_status_after_build(self):
        client.post("/rag/index/build/sync")
        response = client.get("/rag/index/status")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert data["vector_count"] > 0

    def test_rag_query_returns_answer(self):
        client.post("/rag/index/build/sync")
        payload = {
            "question": "What are the most common billing complaints?",
            "product": None,
            "issue": None,
            "date_filter": None,
        }
        response = client.post("/rag/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["answer"]) > 0
        assert isinstance(data["complaint_themes"], list)
        assert isinstance(data["cited_record_ids"], list)

    def test_rag_query_with_filters(self):
        client.post("/rag/index/build/sync")
        payload = {
            "question": "What billing disputes occurred?",
            "product": "Credit card",
            "issue": "Billing",
            "date_filter": "2022-01-01",
        }
        response = client.post("/rag/query", json=payload)
        assert response.status_code == 200


# ── Unified Intel Endpoint ─────────────────────────────────────────────────────

class TestIntelEndpoint:

    def test_customer_intel_full_pipeline(self):
        # Setup: train model + build index
        client.post("/ml/train/sync", json={"retrain": False, "force_promote": True})
        client.post("/rag/index/build/sync")

        payload = {
            "customer_features": SAMPLE_FEATURES,
            "product": "Credit card",
            "issue": "Billing disputes",
            "date_filter": "2021-01-01",
        }
        response = client.post("/customer-intel", json=payload)
        assert response.status_code == 200
        data = response.json()

        # ML outputs
        assert 0.0 <= data["conversion_probability"] <= 1.0
        assert data["conversion_band"] in ("LOW", "MEDIUM", "HIGH")
        assert len(data["feature_importance"]) > 0

        # RAG outputs
        assert isinstance(data["complaint_themes"], list)
        assert isinstance(data["cited_record_ids"], list)
        assert len(data["rag_answer"]) > 0

        # Confidence metrics
        cm = data["confidence_metrics"]
        assert "ml_confidence" in cm
        assert "rag_confidence" in cm
        assert "model_version" in cm

        # Metadata
        assert "request_id" in data
        assert data["processing_time_ms"] > 0

    def test_customer_intel_validates_features(self):
        bad_payload = {
            "customer_features": {**SAMPLE_FEATURES, "credit_score": 100},  # < 300
            "product": None,
            "issue": None,
            "date_filter": None,
        }
        response = client.post("/customer-intel", json=bad_payload)
        assert response.status_code == 422

    def test_monitoring_export_succeeds(self):
        response = client.get("/monitoring/export")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "metrics" in data
        assert "drift_reports_count" in data

    def test_batch_score_json_succeeds(self):
        client.post("/ml/train/sync", json={"retrain": False, "force_promote": True})
        payload = [SAMPLE_FEATURES, SAMPLE_FEATURES]
        response = client.post("/ml/batch-score", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["total_records"] == 2
        assert len(data["predictions"]) == 2
        assert data["predictions"][0]["status"] == "success"

    def test_batch_score_csv_succeeds(self):
        import io
        import csv
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=SAMPLE_FEATURES.keys())
        writer.writeheader()
        writer.writerow(SAMPLE_FEATURES)
        writer.writerow(SAMPLE_FEATURES)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")
        
        files = {"file": ("test.csv", csv_bytes, "text/csv")}
        response = client.post("/ml/batch-score", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["total_records"] == 2
        assert len(data["predictions"]) == 2

    def test_rag_refusal_on_out_of_domain(self):
        client.post("/rag/index/build/sync")
        payload = {
            "question": "How do I bake a chocolate cake?",
            "product": None,
            "issue": None,
            "date_filter": None,
        }
        response = client.post("/rag/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "decline to answer" in data["answer"].lower() or "could not find any relevant" in data["answer"].lower() or "sorry" in data["answer"].lower()
        assert data["confidence_score"] == 0.0
        assert len(data["cited_record_ids"]) == 0
