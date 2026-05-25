# Demo Recording Script & Walkthrough Guide (2–3 Minutes)

This document provides a highly structured script and visual guide to record a perfect, 2-to-3 minute video demonstration of the **Customer Intelligence Platform (CIP)** for submission.

---

## 1. Preparation Checklist

- **Local Setup:** Start all services by running `docker compose up --build` or starting the local backend in the virtualenv (`uvicorn app.main:app --port 8000 --reload`) and opening the frontend on `http://localhost:8080`.
- **Pre-bootstrap:** Ensure the model is trained and the FAISS index is built by triggering bootstrap endpoints or hitting the training/indexing buttons on the dashboard.
- **Recording Tool:** Use a screen recorder with microphone enabled (e.g. **Loom**, **Zoom**, or macOS **QuickTime Player**). Set screen resolution to 1080p.

---

## 2. Audio-Visual Script Timeline

### Phase 1: High-Impact Introduction (0:00 – 0:30)
- **Visual:** Display the main "Intelligence" tab dashboard running on `http://localhost:8080`. Move your mouse slightly over the active tabs to show smooth transition hover states and the animated glowing background orbs.
- **Narrative:**
  > "Hi everyone, welcome to the demonstration of the Customer Intelligence Platform—a production-ready AI system designed for campaign conversion prediction and complaint intelligence. In a single, parallel API call, this platform merges well-calibrated machine learning classifiers with semantic RAG over bank customer complaints, all backed by a robust MLOps pipeline and strict relative governance gates."

### Phase 2: Unified Customer Intelligence in Action (0:30 – 1:15)
- **Visual:** On the left "Customer Profile" card, input:
  - Age: `35`, Credit Score: `720`, Balance: `$75,000`, Salary: `$65,000`, Geography: `France`.
  - In the complaint query, type: *"unexpected late fee charged on my credit card disputing autopay."*
  - Click **Analyze Customer**. Show the sleek loading indicator spinner on the button, then watch the results load dynamically.
- **Narrative:**
  > "Let's run a unified analysis. We input a customer profile alongside their query. In the background, the gateway fans out ML and RAG requests in parallel. Our calibrated GBT classifier outputs an exact conversion probability of 84% in the HIGH band, complete with bootstrap confidence intervals and feature importance importances showing exactly what drove the score. At the same exact time, our LangChain RAG pipeline retrieves the most semantically relevant CFPB complaints, extracts key themes, and renders verified evidence records with source citations."

### Phase 3: Adversarial Robustness & Out-of-Domain Refusal (1:15 – 1:45)
- **Visual:** Keep the same profile but change the complaint query text to: *"What is the capital of France?"*
- **Visual:** Click **Analyze Customer**. Point out the immediate refusal message in the AI Answer box and the `0.00` RAG confidence pill.
- **Narrative:**
  > "CIP is built defensively. If an operator asks a question completely outside the financial complaints domain—like this general knowledge question—our retriever's cosine similarity score falls below our safety gate threshold of 0.35. The RAG service immediately intercept the query in the vector layer and politely refuses to answer, protecting the system against jailbreak attempts and hallucinations."

### Phase 4: MLOps Spine & DevOps Evidence (1:45 – 2:30)
- **Visual:** Toggle to the **ML Service** tab. Show the "Model Info" and "Drift Detection" sections.
- **Visual:** Toggle to the **Monitoring** tab. Show the live service health indicators and links to Prometheus, Grafana, and MLflow.
- **Narrative:**
  > "Finally, let's explore our MLOps spine. We enforce a 7-stage ML pipeline with a relative promotion gate—requiring a 3% PR-AUC improvement to prevent regression before deploying models to the MLflow Registry. Operational metrics are continuously scraped by Prometheus for Grafana dashboards, while EvidentlyAI monitors incoming features against our training reference parquet to detect covariate drift. The entire stack deploys securely onto Azure Static Web Apps and Container Apps via our automated Azure DevOps YAML pipeline."

### Phase 5: Conclusion & Outro (2:30 – 3:00)
- **Visual:** Toggle back to the **Intelligence** tab, hover over the brand logo.
- **Narrative:**
  > "With unified async gateways, well-calibrated classifiers, secure semantic refusal gates, and automated DevOps bootstrapping, CIP demonstrates how modern AI systems can be engineered safely, observably, and highly performantly for production workloads. Thank you!"
