# Production Hardening Plan

This document outlines security measures, input scrubbing, rate limiting, and defensive actions required to secure the Customer Intelligence Platform for production environments.

---

## 1. Input Scrubbing & Schema Validation

- **Pydantic Validation:** All incoming endpoints rely on Pydantic schemas (`CustomerFeatures`, `ComplaintQuery`) which strictly validate types and data bounds (e.g., age `[18, 100]`, credit score `[300, 850]`).
- **CSV Protection:** The `/ml/batch-score` CSV upload endpoint restricts file uploads, strictly parsing rows via Pandas with error isolation. 
- **Recommendation:** Implement a custom upload file size limit (e.g., maximum 10MB) in the FastAPI middleware to prevent Denial of Service (DoS) attacks via massive file uploads.

---

## 2. API Rate Limiting

- **Current State:** The API does not have rate-limiting middleware configured.
- **Hardening Action:** Add standard rate limiting using `slowapi` or an Nginx ingress gateway:
  - Unified endpoint (`/customer-intel`): Limit to 100 requests per minute per IP.
  - Training endpoint (`/ml/train/sync`): Limit to 5 requests per hour per user.
  - Batch scoring (`/ml/batch-score`): Limit to 20 requests per minute per IP.

---

## 3. Defense against Adversarial Prompts (RAG Jailbreaking)

- **Input Sanitization:** Sanitize input strings in the RAG gateway by stripping control characters and restricting string length to `200` characters.
- **Top-1 Score Gate:** The active refusal gateway drops queries whose cosine similarity score falls below `0.35`. This effectively blocks standard out-of-domain jailbreak prompts because their semantic embeddings have low correlation with financial complaints.
- **PII Masking:** Use regular expressions inside `app/rag/pipeline.py` or a dedicated library to scrub potential PII (social security numbers, phone numbers, credit cards) before forwarding text inputs to the LLM backend.

---

## 4. Environment and Dependency Security

- **Secrets Management:** Ensure all keys (`OPENAI_API_KEY`, `GRAFANA_ADMIN_PASSWORD`) are loaded strictly via `.env` or system environment variables, never hardcoded in files.
- **Security Scans:** Configure automated vulnerability scanning using `Snyk` or `GitHub Dependabot` in `.github/workflows/ci.yml` to identify stale or vulnerable packages.
- **Isolated Containers:** Run Docker services as non-root users inside container configs (`docker-compose.yml`) to prevent container escape exploits.
