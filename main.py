"""
RAG API - Main Application
──────────────────────────
FastAPI application connecting all 5 pipelines.

Endpoints:
- POST   /documents/upload          - Upload document
- POST   /documents/{id}/process    - Process document
- GET    /documents/{id}/status     - Get processing status
- POST   /collections/              - Create collection
- GET    /collections/              - List collections
- GET    /collections/{name}        - Get collection info
- DELETE /collections/{name}        - Delete collection
- POST   /query/                    - Query RAG system
- GET    /query/conversations/{id}  - Get conversation history
- DELETE /query/conversations/{id}  - Clear conversation
- GET    /health                    - Health check
- GET    /status                    - System status
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config.settings import RAGConfig, get_pipeline_manager
from .routers import documents, query, collections
from .schemas.models import HealthResponse, SystemStatusResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    """
    # Startup
    log.info("Starting RAG API...")
    
    # Validate configuration
    config_errors = RAGConfig.validate()
    if config_errors:
        log.error(f"Configuration errors: {config_errors}")
        raise ValueError(f"Invalid configuration: {config_errors}")
    
    # Initialize pipelines
    pm = get_pipeline_manager()
    
    try:
        pm.initialize_all()
        log.info("✓ All pipelines initialized successfully")
    except Exception as e:
        log.error(f"Failed to initialize pipelines: {e}")
        raise
    
    log.info(f"RAG API started on {RAGConfig.API_HOST}:{RAGConfig.API_PORT}")
    
    yield
    
    # Shutdown
    log.info("Shutting down RAG API...")


# Create FastAPI app
app = FastAPI(
    title="Legal RAG API",
    description="Production-ready RAG system for legal documents",
    version=RAGConfig.API_VERSION,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router)
app.include_router(collections.router)
app.include_router(query.router)


# ══════════════════════════════════════════════════════════════════════════
# Health & Status Endpoints
# ══════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint."""
    return {
        "message": "Legal RAG API",
        "version": RAGConfig.API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns status of all pipelines.
    """
    pm = get_pipeline_manager()
    pipeline_status = pm.health_check()
    
    all_healthy = all(pipeline_status.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version=RAGConfig.API_VERSION,
        pipelines=pipeline_status,
        message="All systems operational" if all_healthy else "Some pipelines unavailable",
    )


@app.get("/status", response_model=SystemStatusResponse, tags=["Health"])
async def system_status():
    """
    Detailed system status.
    """
    pm = get_pipeline_manager()
    pipeline_status = pm.health_check()
    
    # Get collection stats
    try:
        collections = pm.storage.list_collections()
        total_collections = len(collections)
        total_documents = sum(c.num_documents for c in collections)
    except:
        total_collections = 0
        total_documents = 0
    
    return SystemStatusResponse(
        api_version=RAGConfig.API_VERSION,
        ingestion_ready=pipeline_status["ingestion"],
        chunker_ready=pipeline_status["chunker"],
        embedder_ready=pipeline_status["embedder"],
        storage_ready=pipeline_status["storage"],
        llm_ready=pipeline_status["llm"],
        total_collections=total_collections,
        total_documents=total_documents,
        embedding_model=RAGConfig.EMBEDDING_MODEL,
        embedding_dim=RAGConfig.EMBEDDING_DIM,
        llm_model=RAGConfig.LLM_MODEL,
    )


# ══════════════════════════════════════════════════════════════════════════
# Exception Handlers
# ══════════════════════════════════════════════════════════════════════════

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    log.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "status_code": 500,
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "rag_api.main:app",
        host=RAGConfig.API_HOST,
        port=RAGConfig.API_PORT,
        reload=True,
        log_level="info",
    )