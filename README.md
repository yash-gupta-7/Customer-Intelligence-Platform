# Customer Intelligence Platform

Production-grade AI platform combining **ML campaign conversion prediction** and **RAG-based complaint intelligence**, with an MLOps spine for monitoring, drift detection, and model governance.

## Features

- **ML Service** — 7-stage pipeline: ingest → validate → features → train → evaluate → relative gate → serve
- **RAG Service** — 8-stage complaint intelligence with FAISS retrieval and grounded answer synthesis
- **Observability** — Prometheus metrics, EvidentlyAI drift reports, structured telemetry export
- **Dashboard** — Single-page UI for predictions, RAG queries, and system health

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

API: `http://localhost:8000` · Docs: `http://localhost:8000/docs` · UI: `http://localhost:8080`

## Project layout

```
├── backend/          # FastAPI service (ML, RAG, monitoring)
├── frontend/         # Dashboard UI
├── infra/            # Nginx, Prometheus configs
├── docs/             # Architecture, decisions, hardening plan
└── docker-compose.yml
```

## Documentation

- [Architecture](docs/architecture.md)
- [Decision log](docs/decision_log.md)
- [Hardening plan](docs/hardening_plan.md)

## License

MIT — see [LICENSE](LICENSE).
