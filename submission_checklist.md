# MLOps & AI Engineering Project Submission Checklist

This file serves as the master entrypoint for the final project submission, linking all required deliverables, reports, and evidence.

---

## 1. Primary Repository Coordinates

- **GitHub Repository:** [github.com/yash-gupta-7/Customer-Intelligence-Platform](https://github.com/yash-gupta-7/Customer-Intelligence-Platform.git)
- **Azure DevOps Repository:** [dev.azure.com/yashgupta0078/Customer](https://yashgupta0078@dev.azure.com/yashgupta0078/Customer/_git/Customer)
- **CI/CD Pipeline History:** [dev.azure.com/yashgupta0078/Customer/_build](https://dev.azure.com/yashgupta0078/Customer/_build)

---

## 2. Core Submission Deliverables

We have compiled highly detailed, production-grade reports for every requirement in **Section 11 (Submission and Sequence)** and **Section 12 (Reflection)**:

### 📑 [Model Evaluation Report](file:///Users/yash/Customer%20Intelligence%20Platform/docs/model_report.md)
- **Contents:** Well-calibrated conversion metrics (AUC-ROC, PR-AUC, Accuracy, Precision, Recall, F1), optimal decision thresholds (`0.50`), detailed Confusion Matrix, and historical Relative Promotion Gate audit logs (`logs/promotion.log`) proving regression defense.

### 🔍 [RAG Evaluation Report](file:///Users/yash/Customer%20Intelligence%20Platform/docs/rag_report.md)
- **Contents:** langChain RAG configuration (`all-MiniLM-L6-v2` dense embeddings, local FAISS Flat IP index), sample questions from the custom evaluation harness (`backend/app/rag/eval_harness.py`), retrieved evidence logs, and zero-hallucination refusal gates protecting against adversarial prompt injections.

### 📊 [Operations & Monitoring Report](file:///Users/yash/Customer%20Intelligence%20Platform/docs/monitoring_report.md)
- **Contents:** Custom Prometheus metrics configurations, API Gateway `/monitoring/export` telemetry JSON schemas, and EvidentlyAI Kolmogorov-Smirnov statistical covariate drift checks comparing production inputs against reference parquet baselines.

### ☁️ [Deployment & IaC Evidence](file:///Users/yash/Customer%20Intelligence%20Platform/docs/deployment_evidence.md)
- **Contents:** Cloud deployment topology (Azure Static Web Apps for frontend, Azure Container Apps for serverless API, Azure Container Registry, persistent storage mounts), Azure Bicep IaC details, and 4-stage DevOps CI/CD pipeline history (`azure-pipelines.yml`) detailing automated post-deploy model training and RAG indexing.

### 🎥 [2–3 Minute Demo Recording Script](file:///Users/yash/Customer%20Intelligence%20Platform/docs/demo_recording_script.md)
- **Contents:** High-impact audio-visual walkthrough script tailored to demonstrate unified parallel queries, active micro-animations, semantic out-of-domain refusals, and drift alerts on Loom or QuickTime in exactly 2-3 minutes.

### 🧠 [Engineering Reflection (Section 12)](file:///Users/yash/Customer%20Intelligence%20Platform/reflection.md)
- **Contents:** Exhaustive retrospective on architectural trade-offs, Sigmoid vs Isotonic calibration, metrics serialization limits, similarity-based refusal vs prompt tuning, unified fan-out asyncio latency profiling, and continuous integration governance.
