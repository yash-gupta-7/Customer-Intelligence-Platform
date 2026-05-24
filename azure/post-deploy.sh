#!/usr/bin/env bash
# Post-deploy: bootstrap ML model + FAISS index on the running Container App.
set -euo pipefail

API_URL="${1:?Usage: post-deploy.sh <https://api-fqdn> [api-secret-key]}"
API_SECRET="${2:-${API_SECRET_KEY:-}}"

if [[ -z "$API_SECRET" ]]; then
  echo "ERROR: Set API_SECRET_KEY or pass the secret as the second argument."
  exit 1
fi

API_URL="${API_URL%/}"
echo "==> Waiting for API health at ${API_URL}/health"
for i in $(seq 1 60); do
  if curl -sf "${API_URL}/health" >/dev/null; then
    echo "    API is up."
    break
  fi
  if [[ "$i" -eq 60 ]]; then
    echo "ERROR: API did not become healthy in time."
    exit 1
  fi
  sleep 10
done

echo "==> Training ML model (sync)…"
curl -sf -X POST "${API_URL}/ml/train/sync" \
  -H "Content-Type: application/json" \
  -d '{"retrain": true, "force_promote": true}' \
  --max-time 1800 || {
  echo "WARN: ML train request failed or timed out — run manually later."
}

echo "==> Building RAG index (sync)…"
curl -sf -X POST "${API_URL}/rag/index/build/sync" \
  --max-time 1800 || {
  echo "WARN: RAG index build failed or timed out — run manually later."
}

echo "==> Smoke test (predict)"
curl -sf -X POST "${API_URL}/ml/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_SECRET}" \
  -d '{
    "age": 35,
    "income": 75000,
    "credit_score": 720,
    "account_balance": 12000,
    "num_products": 2,
    "tenure_months": 24,
    "campaign_channel": "email",
    "campaign_type": "promotional",
    "days_since_last_contact": 14,
    "previous_campaign_response": 0,
    "region": "north",
    "employment_status": "employed",
    "has_mortgage": 0,
    "has_credit_card": 1
  }' | head -c 500
echo ""
echo "==> Post-deploy complete."
