"""
API Schemas
───────────
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


# ══════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryTypeEnum(str, Enum):
    NEW = "new"
    FOLLOWUP = "followup"
    CLARIFICATION = "clarification"


class ContextQualityEnum(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NO_CONTEXT = "none"


# ══════════════════════════════════════════════════════════════════════════
# Document Upload & Processing
# ══════════════════════════════════════════════════════════════════════════

class DocumentUploadResponse(BaseModel):
    """Response after uploading a document."""
    document_id: str
    filename: str
    status: DocumentStatus
    message: str


class DocumentProcessRequest(BaseModel):
    """Request to process an uploaded document."""
    collection_name: str = Field(..., description="Target collection name")
    model_version: Optional[str] = Field("v1.0", description="Embedding model version")


class DocumentProcessResponse(BaseModel):
    """Response after document processing."""
    document_id: str
    filename: str
    status: DocumentStatus
    
    # Processing stats
    total_pages: int
    native_pages: int
    ocr_pages: int
    total_chunks: int
    chunks_indexed: int
    
    # Performance
    processing_time_sec: float
    
    # Errors
    errors: List[str] = []


class DocumentStatusResponse(BaseModel):
    """Document processing status."""
    document_id: str
    filename: str
    status: DocumentStatus
    progress_percentage: int
    current_stage: str
    message: str


# ══════════════════════════════════════════════════════════════════════════
# Collection Management
# ══════════════════════════════════════════════════════════════════════════

class CollectionCreateRequest(BaseModel):
    """Request to create a collection."""
    name: str = Field(..., description="Collection name")
    embedding_dim: int = Field(1024, description="Embedding dimension")
    model_name: str = Field("BAAI/bge-large-en-v1.5", description="Embedding model")
    drop_if_exists: bool = Field(False, description="Drop existing collection")


class CollectionCreateResponse(BaseModel):
    """Response after creating collection."""
    name: str
    embedding_dim: int
    model_name: str
    created: bool
    message: str


class CollectionInfo(BaseModel):
    """Collection information."""
    name: str
    num_documents: int
    embedding_dim: int


class CollectionListResponse(BaseModel):
    """List of collections."""
    collections: List[CollectionInfo]
    total: int


# ══════════════════════════════════════════════════════════════════════════
# Query & Search
# ══════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    """Request to query the RAG system."""
    query: str = Field(..., description="User question")
    collection_name: str = Field(..., description="Collection to search")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for multi-turn")
    k: Optional[int] = Field(5, description="Number of contexts to retrieve")
    filter_by: Optional[str] = Field(None, description="Typesense filter expression")


class CitationSchema(BaseModel):
    """Citation information."""
    source_file: str
    hierarchy: List[str]
    page_numbers: List[int]
    score: float


class QueryResponse(BaseModel):
    """Response from RAG query."""
    answer: str
    query: str
    conversation_id: str
    
    # Query metadata
    query_type: QueryTypeEnum
    context_quality: ContextQualityEnum
    
    # Citations
    citations: List[str]
    contexts_used: int
    best_score: float
    
    # Performance
    response_time_sec: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    # Model info
    model_name: str


class ConversationHistoryResponse(BaseModel):
    """Conversation history."""
    conversation_id: str
    messages: List[Dict[str, str]]
    total_messages: int


# ══════════════════════════════════════════════════════════════════════════
# Health & Status
# ══════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """API health status."""
    status: str
    version: str
    pipelines: Dict[str, bool]
    message: str


class SystemStatusResponse(BaseModel):
    """Detailed system status."""
    api_version: str
    
    # Pipeline status
    ingestion_ready: bool
    chunker_ready: bool
    embedder_ready: bool
    storage_ready: bool
    llm_ready: bool
    
    # Storage info
    total_collections: int
    total_documents: int
    
    # Model info
    embedding_model: str
    embedding_dim: int
    llm_model: str


# ══════════════════════════════════════════════════════════════════════════
# Error Responses
# ══════════════════════════════════════════════════════════════════════════

class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str
    status_code: int