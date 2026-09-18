"""
api/main.py

FastAPI application — all endpoints.

Endpoints:
  POST /chat/stream   — streaming SSE answer (answer mode)
  POST /chat          — full response with eval metrics (eval mode)
  POST /ingest        — trigger PDF ingestion (admin only)
  GET  /health        — system health check
  GET  /collections   — Qdrant collection stats
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import API_SECRET_TOKEN, EXTRACTED_IMAGES_DIR, QDRANT_COLLECTIONS
from config.model_loader import warmup_models
from config.qdrant_manager import get_qdrant_client, init_collections
from models.schemas import (
    AppMode,
    ChatRequest,
    ChatResponse,
    Citation,
    ConfidenceLevel,
    EvalMetrics,
    HealthResponse,
    IngestRequest,
    MedRAGState,
)
from rag.pipeline import run_rag_pipeline, run_rag_streaming

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ─── Auth ─────────────────────────────────────────────────────────────────────
security = HTTPBearer(auto_error=False)

def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> None:
    if not credentials or credentials.credentials != API_SECRET_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token.",
        )


# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:
      1. Initialise Qdrant collections (idempotent)
      2. Warm up all local models (download once, cache to disk)
    Shutdown:
      - Nothing special needed (Docker handles cleanup)
    """
    logger.info("MedRAG startup — initialising Qdrant collections...")
    try:
        init_collections()
    except Exception as e:
        logger.error(
            "Qdrant init failed at startup: %s. "
            "Start Qdrant (docker compose up qdrant -d) or set QDRANT_MODE=local.",
            e,
        )
        raise

    logger.info("Warming up local models (embedder, reranker, CLIP)...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, warmup_models)

    logger.info("MedRAG is ready.")
    yield
    logger.info("MedRAG shutdown.")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MedRAG API",
    description="Exam-grounded medical RAG chatbot for DHA, MDS, ORE",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 1 — Streaming chat (Answer mode)
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/chat/stream",
    summary="Streaming SSE chat — Answer mode",
    dependencies=[Depends(verify_token)],
)
async def chat_stream(request: ChatRequest):
    """
    Streams the LLM response token-by-token via Server-Sent Events.
    Events emitted: token | citations | confidence | images | done | error

    Frontend should use EventSource or fetch with SSE parsing.
    """
    if request.mode == AppMode.EVAL:
        raise HTTPException(
            status_code=400,
            detail="Use POST /chat for eval mode (non-streaming).",
        )

    async def event_generator():
        try:
            async for event in run_rag_streaming(
                query=request.query,
                exam=request.exam,
                mode=request.mode,
                history=request.history,
                session_id=request.session_id,
            ):
                yield event
        except Exception as e:
            import json
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 2 — Full chat with eval metrics (Eval mode)
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Full chat with optional RAGAS evaluation",
    dependencies=[Depends(verify_token)],
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Non-streaming endpoint. Returns complete structured response.
    In eval mode, also returns RAGAS metrics (faithfulness, relevancy, etc.)
    Use this for eval mode or when the frontend needs the full JSON at once.
    """
    try:
        state: MedRAGState = await run_rag_pipeline(
            query=request.query,
            exam=request.exam,
            mode=request.mode,
            history=request.history,
            session_id=request.session_id,
        )
    except Exception as e:
        logger.error("Chat pipeline error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e

    # RAGAS evaluation (eval mode only)
    eval_metrics = None
    if request.mode == AppMode.EVAL and state.retrieved_docs:
        try:
            from evaluation.rag_evaluator import run_ragas_evaluation
        except ImportError as e:
            logger.error("Eval mode requires ragas and datasets: %s", e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Eval mode requires: pip install ragas datasets",
            ) from e

        ragas_scores = run_ragas_evaluation(
            query=request.query,
            answer=state.answer,
            retrieved_docs=state.retrieved_docs,
        )
        eval_metrics = EvalMetrics(
            faithfulness=ragas_scores.get("faithfulness"),
            answer_relevancy=ragas_scores.get("answer_relevancy"),
            context_precision=ragas_scores.get("context_precision"),
            context_recall=ragas_scores.get("context_recall"),
            hallucination_flag=state.hallucination_grade == "hallucinated",
        )

    # Confidence level
    if state.confidence >= 0.85:
        conf_level = ConfidenceLevel.HIGH
    elif state.confidence >= 0.70:
        conf_level = ConfidenceLevel.MEDIUM
    else:
        conf_level = ConfidenceLevel.LOW

    # Build citations
    citations = [
        Citation(
            book_name=c["book_name"],
            chapter=c["chapter"],
            section=c["section"],
            page_number=c["page_number"],
            chunk_type=c["chunk_type"],
            has_image=c["has_image"],
            image_path=c.get("image_path"),
            relevance_score=c["relevance_score"],
        )
        for c in state.citations
    ]

    return ChatResponse(
        answer=state.answer,
        exam=state.exam,
        mode=request.mode,
        query_type=state.query_type,
        confidence=state.confidence,
        confidence_level=conf_level,
        citations=citations,
        has_images=state.has_images,
        image_paths=state.image_paths,
        eval_metrics=eval_metrics,
        retry_count=state.retry_count,
        session_id=request.session_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 3 — Ingestion trigger (admin)
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/ingest",
    summary="Trigger PDF ingestion for an exam (admin only)",
)
async def ingest(request: IngestRequest):
    """
    Triggers the full ingestion pipeline for the specified exam.
    Runs in background — returns immediately with a status message.
    Only callable with admin token (API_SECRET_TOKEN).
    """
    if request.admin_token != API_SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token.")

    from ingestion.pdf_ingester import ingest_exam

    async def run_ingestion():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, ingest_exam, request.exam.value)

    asyncio.create_task(run_ingestion())

    return {
        "status": "started",
        "message": f"Ingestion started for exam '{request.exam}' in background.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 4 — Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
)
async def health() -> HealthResponse:
    """Returns system status — useful for Docker health checks."""
    qdrant_ok = False
    try:
        client = get_qdrant_client()
        client.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    models_ok = False
    try:
        from config.model_loader import get_embedder, get_reranker

        get_embedder()
        get_reranker()
        models_ok = True
    except Exception:
        pass

    from config.qdrant_manager import resolve_qdrant_mode

    # Embedded/local Qdrant is valid; only require models for "ok"
    status = "ok" if qdrant_ok and models_ok else "degraded"
    if resolve_qdrant_mode() == "local" and models_ok:
        status = "ok"

    return HealthResponse(
        status=status,
        qdrant_connected=qdrant_ok,
        models_loaded=models_ok,
        exams_available=list(QDRANT_COLLECTIONS.keys()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 5 — Collection stats
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/images/{file_path:path}",
    summary="Serve extracted medical images",
    dependencies=[Depends(verify_token)],
)
async def serve_image(file_path: str):
    """Serve images from extracted_images/ (used by frontend lightbox)."""
    base = EXTRACTED_IMAGES_DIR.resolve()
    target = (EXTRACTED_IMAGES_DIR / file_path).resolve()
    if not str(target).startswith(str(base)) or not target.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    media = "image/png" if target.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path=target, media_type=media)


@app.get(
    "/collections",
    summary="Qdrant collection statistics",
    dependencies=[Depends(verify_token)],
)
async def collection_stats():
    """Returns point counts and vector info for each exam collection."""
    client = get_qdrant_client()
    stats = {}
    for exam, name in QDRANT_COLLECTIONS.items():
        try:
            info = client.get_collection(name)
            stats[exam] = {
                "collection": name,
                "vectors_count": info.vectors_count,
                "points_count":  info.points_count,
                "status":        info.status,
            }
        except Exception as e:
            stats[exam] = {"error": str(e)}
    return stats