"""
main.py — FastAPI application entrypoint.
"""
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings
from app.monitoring.metrics import setup_logging
from app.routers import ml_router, rag_router, intel_router, monitor_router

settings = get_settings()


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup: create dirs, pre-load models, build RAG index."""
    setup_logging(settings.log_level)
    logger.info(f"Starting {settings.app_name} v{settings.app_version} [{settings.app_env}]")

    # Ensure directories exist
    for path in [
        "data/raw", "data/processed", "data/complaints",
        "models", "reports/drift", "logs", "mlruns",
    ]:
        os.makedirs(path, exist_ok=True)

    # Eagerly load ML model (if already trained)
    try:
        from app.ml.model import get_model
        model = get_model(settings.model_path)
        if model.model is not None:
            logger.info(f"ML model pre-loaded: version={model.version}")
        else:
            logger.warning("No trained ML model found. Call POST /ml/train/sync to train.")
    except Exception as e:
        logger.warning(f"ML model pre-load skipped: {e}")

    # Eagerly load RAG index (if already built)
    try:
        from app.rag.retriever import get_retriever
        from app.monitoring.metrics import INDEX_SIZE
        retriever = get_retriever()
        if retriever.index and retriever.index.ntotal > 0:
            INDEX_SIZE.set(retriever.index.ntotal)
            logger.info(f"FAISS index pre-loaded: {retriever.index.ntotal} vectors")
        else:
            logger.warning("No FAISS index found. Call POST /rag/index/build/sync to build.")
    except Exception as e:
        logger.warning(f"FAISS pre-load skipped: {e}")

    logger.info("Application ready ✓")
    yield
    logger.info("Application shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Intelligence Platform",
    description=(
        "Production-grade AI platform combining ML conversion prediction "
        "and RAG-based complaint intelligence with a shared MLOps spine."
    ),
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    import uuid
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(elapsed)
    return response


# ── Prometheus instrumentation ────────────────────────────────────────────────

Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/health", "/metrics"],
).instrument(app).expose(app, endpoint="/metrics")


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(ml_router.router)
app.include_router(rag_router.router)
app.include_router(intel_router.router)
app.include_router(monitor_router.router)


# ── Health & Info endpoints ───────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": settings.app_version, "env": settings.app_env}


@app.get("/", tags=["Info"])
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "endpoints": {
            "unified": "POST /customer-intel",
            "ml_predict": "POST /ml/predict",
            "ml_train": "POST /ml/train/sync",
            "ml_drift": "POST /ml/drift",
            "rag_query": "POST /rag/query",
            "rag_build": "POST /rag/index/build/sync",
        },
    }


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "path": str(request.url)},
    )
