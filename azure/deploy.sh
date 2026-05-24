#!/usr/bin/env bash
# Provision Azure resources, build the API image in ACR, deploy Container App, bootstrap models.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-cip-prod}"
LOCATION="${LOCATION:-eastus}"
BASE_NAME="${BASE_NAME:-cip}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
API_SECRET_KEY="${API_SECRET_KEY:-}"

if ! command -v az >/dev/null 2>&1; then
  echo "ERROR: Azure CLI (az) is required. Install: https://learn.microsoft.com/cli/azure/install-azure-cli"
  exit 1
fi

az account show >/dev/null 2>&1 || {
  echo "ERROR: Run 'az login' first."
  exit 1
}

if [[ -z "$API_SECRET_KEY" ]]; then
  if command -v openssl >/dev/null 2>&1; then
    API_SECRET_KEY="$(openssl rand -hex 32)"
    echo "==> Generated API_SECRET_KEY (save this): ${API_SECRET_KEY}"
  else
    echo "ERROR: Set API_SECRET_KEY in the environment."
    exit 1
  fi
fi

echo "==> Ensuring resource group: ${RESOURCE_GROUP}"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

echo "==> Deploying base infrastructure (ACR, logs, environment)…"
DEPLOY_OUT="$(az deployment group create \
  --resource-group "$RESOURCE_GROUP" \
  --template-file azure/main.bicep \
  --parameters baseName="$BASE_NAME" \
    containerImage="mcr.microsoft.com/k8se/quickstart:latest" \
    apiSecretKey="$API_SECRET_KEY" \
    allowedOrigins="${ALLOWED_ORIGINS:-*}" \
    deployStaticWebApp="${DEPLOY_STATIC_WEB_APP:-true}" \
    deployApi=false \
  --query properties.outputs -o json)"

ACR_NAME="$(echo "$DEPLOY_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['acrName']['value'])")"
ACR_LOGIN="$(echo "$DEPLOY_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['acrLoginServer']['value'])")"
FULL_IMAGE="${ACR_LOGIN}/cip-api:${IMAGE_TAG}"

echo "==> Building and pushing image in ACR: ${FULL_IMAGE}"
az acr build \
  --registry "$ACR_NAME" \
  --image "cip-api:${IMAGE_TAG}" \
  --file backend/Dockerfile \
  backend

echo "==> Deploying Container App with application image…"
az deployment group create \
  --resource-group "$RESOURCE_GROUP" \
  --template-file azure/main.bicep \
  --parameters baseName="$BASE_NAME" \
    containerImage="$FULL_IMAGE" \
    apiSecretKey="$API_SECRET_KEY" \
    allowedOrigins="${ALLOWED_ORIGINS:-*}" \
    deployStaticWebApp="${DEPLOY_STATIC_WEB_APP:-true}" \
    deployApi=true \
  --output none

FQDN="$(az containerapp show --name "${BASE_NAME}-api" --resource-group "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv)"
API_URL="https://${FQDN}"

echo ""
echo "==> API URL: ${API_URL}"
echo "==> API_SECRET_KEY: ${API_SECRET_KEY}"
echo ""

chmod +x azure/post-deploy.sh
API_SECRET_KEY="$API_SECRET_KEY" azure/post-deploy.sh "$API_URL" "$API_SECRET_KEY"

SWA_HOST="$(az staticwebapp show --name "${BASE_NAME}-web" --resource-group "$RESOURCE_GROUP" --query defaultHostname -o tsv 2>/dev/null || true)"
if [[ -n "$SWA_HOST" ]]; then
  echo ""
  echo "==> Static Web App: https://${SWA_HOST}"
  echo "    Deploy frontend with pipeline job or SWA CLI (see azure/README.md)."
fi

echo ""
echo "==> Deployment finished."
