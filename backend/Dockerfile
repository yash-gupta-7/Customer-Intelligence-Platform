# ════════════════════════════════════════════════════════════════
# Customer Intelligence Platform — Backend Dockerfile
# Multi-stage build for lean production image
# ════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for ML/numeric packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install from pre-built wheels (fast, no compilation)
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Copy application code and sample complaint data for RAG indexing
COPY app/ ./app/
COPY data/complaints/ ./data/complaints/
COPY scripts/ ./scripts/

# Create runtime directories
RUN mkdir -p data/raw data/processed data/complaints \
    models reports/drift logs mlruns

# Non-root user for security
RUN useradd -m -u 1001 appuser && \
    chmod +x /app/scripts/docker-entrypoint.sh && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Use exec form to receive signals properly
ENV UVICORN_WORKERS=1
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
