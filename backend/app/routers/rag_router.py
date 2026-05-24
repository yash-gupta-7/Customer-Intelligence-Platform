"""
routers/rag_router.py — RAG / complaint intelligence endpoints.
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from loguru import logger

from app.auth import require_api_key
from app.config import get_settings, Settings
from app.rate_limit import limiter
from app.rag.pipeline import build_rag_index, run_rag_query
from app.rag.retriever import get_retriever
from app.schemas.rag_schema import ComplaintQuery, RAGResponse, IndexBuildResponse
from app.monitoring.metrics import record_retrieval, INDEX_SIZE

router = APIRouter(prefix="/rag", tags=["RAG Service"])


@router.post(
    "/query",
    response_model=RAGResponse,
    summary="Query complaint intelligence",
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("15/minute")
async def query_complaints(
    request: Request,
    query: ComplaintQuery,
    settings: Settings = Depends(get_settings),
):
    """
    Answer a natural-language question about customer complaints.
    Retrieves grounded evidence from the FAISS index and cites record IDs.
    """
    retriever = get_retriever()
    if retriever.index is None or retriever.index.ntotal == 0:
        raise HTTPException(
            status_code=503,
            detail="RAG index not built. Call POST /rag/index/build first.",
        )
    try:
        response = run_rag_query(
            question=query.question,
            product=query.product,
            issue=query.issue,
            date_filter=query.date_filter,
            top_k=query.top_k,
        )
        record_retrieval(response.retrieval_count, response.confidence_score)
        return response
    except Exception as e:
        logger.exception(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index/build", response_model=IndexBuildResponse, summary="Build FAISS complaint index")
async def build_index(
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
):
    """
    Trigger the full RAG indexing pipeline (stages 1–5) asynchronously.
    Load → Clean → Chunk → Embed → Index.
    """
    def _build():
        try:
            result = build_rag_index(settings.complaints_data_path)
            retriever = get_retriever()
            if retriever.index:
                INDEX_SIZE.set(retriever.index.ntotal)
            logger.info(f"Index built: {result}")
        except Exception as e:
            logger.exception(f"Background index build failed: {e}")

    background_tasks.add_task(_build)
    return IndexBuildResponse(
        status="building",
        documents_indexed=0,
        index_path=settings.faiss_index_path,
        message="Index build started in background. Poll /rag/index/status to check progress.",
    )


@router.post("/index/build/sync", response_model=IndexBuildResponse, summary="Build index synchronously")
async def build_index_sync(settings: Settings = Depends(get_settings)):
    """Build the FAISS index synchronously. Blocks until complete."""
    try:
        result = build_rag_index(settings.complaints_data_path)
        retriever = get_retriever()
        if retriever.index:
            INDEX_SIZE.set(retriever.index.ntotal)
        return IndexBuildResponse(**result)
    except Exception as e:
        logger.exception(f"Index build failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/index/status", summary="Get FAISS index status")
async def index_status():
    """Return current FAISS index size and readiness."""
    retriever = get_retriever()
    ntotal = retriever.index.ntotal if retriever.index else 0
    return {
        "ready": ntotal > 0,
        "vector_count": ntotal,
        "document_count": len(retriever.documents),
        "index_path": retriever.index_path,
    }
