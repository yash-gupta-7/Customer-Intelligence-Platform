# Azure deployment — Customer Intelligence Platform

Deploys:

- **Azure Container Registry** + cloud build (`az acr build`)
- **Azure Container Apps** — FastAPI API (`cip-api`)
- **Azure Static Web Apps** (optional) — static frontend
- **Post-deploy bootstrap** — ML train + FAISS index build

## One-command deploy (local CLI)

```bash
az login
az account set --subscription "<your-subscription>"

export API_SECRET_KEY="$(openssl rand -hex 32)"   # save this
export ALLOWED_ORIGINS="*"                        # tighten after SWA URL is known

chmod +x azure/deploy.sh azure/post-deploy.sh
./azure/deploy.sh
```

Outputs: API URL, `API_SECRET_KEY`, Static Web App hostname.

## Azure DevOps pipeline

1. **Project settings → Service connections** → New **Azure Resource Manager** → name: `azure-cip-subscription`.

2. **Pipelines → Library → Variable group** `cip-azure-prod`:

   | Variable | Secret | Example |
   |----------|--------|---------|
   | `API_SECRET_KEY` | Yes | long random string |
   | `ALLOWED_ORIGINS` | No | `https://cip-web.azurestaticapps.net,http://localhost:8080` |
   | `AZURE_STATIC_WEB_APPS_API_TOKEN` | Yes | from Static Web App deployment token (optional) |

3. **Pipelines → New pipeline** → Azure Repos → `azure-pipelines.yml`.

4. Run pipeline on `main`.

## After deploy

```bash
API_URL="https://<your-fqdn>"
KEY="<API_SECRET_KEY>"

curl "$API_URL/health"
curl -X POST "$API_URL/ml/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $KEY" \
  -d @- <<'JSON'
{"age":35,"income":75000,"credit_score":720,"account_balance":12000,"num_products":2,"tenure_months":24,"campaign_channel":"email","campaign_type":"promotional","days_since_last_contact":14,"previous_campaign_response":0,"region":"north","employment_status":"employed","has_mortgage":0,"has_credit_card":1}
JSON
```

Update `frontend/staticwebapp.config.json` — replace `REPLACE_WITH_CONTAINER_APP_FQDN` with your Container App FQDN (no `https://`).

Get SWA deployment token:

```bash
az staticwebapp secrets list --name cip-web --resource-group rg-cip-prod --query properties.apiKey -o tsv
```

## Resource names (defaults)

| Resource | Name |
|----------|------|
| Resource group | `rg-cip-prod` |
| Container App | `cip-api` |
| Static Web App | `cip-web` |
| ACR | `cipacr<unique>` |

## Troubleshooting

- **Disk / build failures locally** — use `az acr build` (runs in Azure, not on your Mac).
- **502 after deploy** — bootstrap can take 10–20 min; check logs:  
  `az containerapp logs show --name cip-api --resource-group rg-cip-prod --follow`
- **CORS errors** — set `ALLOWED_ORIGINS` to your Static Web App URL and redeploy.
