"""
Configuration & Pipeline Initialization
────────────────────────────────────────
Loads config from environment and initializes all 5 pipelines.
"""

import os
import logging
from typing import Optional

# Pipeline imports
import sys
sys.path.insert(0, "/home/claude")

from doc_pipeline import DocumentIngestionPipeline
from legal_rag_chunker import HierarchicalChunkingPipeline
from embedding_pipeline import EmbeddingPipeline, EmbeddingConfig
from vector_storage import VectorStoragePipeline, StorageConfig
from llm_pipeline import LLMPipeline, LLMConfig

log = logging.getLogger(__name__)


class RAGConfig:
    """
    Configuration for the complete RAG system.
    Loads from environment variables with sensible defaults.
    """
    
    # API
    API_VERSION: str = "1.0.0"
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Embedding
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    EMBEDDING_FALLBACK: str = os.getenv("EMBEDDING_FALLBACK", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1024"))
    
    # Vector Storage (Typesense)
    TYPESENSE_HOST: str = os.getenv("TYPESENSE_HOST", "localhost")
    TYPESENSE_PORT: int = int(os.getenv("TYPESENSE_PORT", "8108"))
    TYPESENSE_PROTOCOL: str = os.getenv("TYPESENSE_PROTOCOL", "http")
    TYPESENSE_API_KEY: str = os.getenv("TYPESENSE_API_KEY", "xyz")
    
    # LLM (Groq)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    
    # Processing
    OCR_DPI: int = int(os.getenv("OCR_DPI", "300"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))
    CONTEXT_THRESHOLD: float = float(os.getenv("CONTEXT_THRESHOLD", "0.3"))
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "5"))
    
    # File storage
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/tmp/rag_uploads")
    
    @classmethod
    def validate(cls) -> list[str]:
        """Validate configuration."""
        errors = []
        
        if not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is required")
        
        if not cls.TYPESENSE_API_KEY:
            errors.append("TYPESENSE_API_KEY is required")
        
        return errors


class PipelineManager:
    """
    Manages all 5 pipelines as singletons.
    Initializes once at startup.
    """
    
    _instance: Optional['PipelineManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        
        # Pipeline instances
        self.ingestion: Optional[DocumentIngestionPipeline] = None
        self.chunker: Optional[HierarchicalChunkingPipeline] = None
        self.embedder: Optional[EmbeddingPipeline] = None
        self.storage: Optional[VectorStoragePipeline] = None
        self.llm: Optional[LLMPipeline] = None
        
        # Conversation storage (in-memory for now)
        self.conversations: dict = {}  # conversation_id -> history
        
        log.info("PipelineManager created")
    
    def initialize_all(self):
        """Initialize all pipelines."""
        log.info("Initializing RAG pipelines...")
        
        try:
            # Stage 1: Document Ingestion
            self.ingestion = DocumentIngestionPipeline(
                ocr_dpi=RAGConfig.OCR_DPI
            )
            log.info("✓ Document Ingestion Pipeline initialized")
            
            # Stage 2: Hierarchical Chunking
            self.chunker = HierarchicalChunkingPipeline()
            log.info("✓ Hierarchical Chunking Pipeline initialized")
            
            # Stage 3: Embedding Generation
            embedding_config = EmbeddingConfig(
                model_name=RAGConfig.EMBEDDING_MODEL,
                fallback_model=RAGConfig.EMBEDDING_FALLBACK,
                batch_size=RAGConfig.BATCH_SIZE,
            )
            self.embedder = EmbeddingPipeline(embedding_config)
            log.info(f"✓ Embedding Pipeline initialized ({RAGConfig.EMBEDDING_MODEL})")
            
            # Stage 4: Vector Storage
            storage_config = StorageConfig(
                host=RAGConfig.TYPESENSE_HOST,
                port=RAGConfig.TYPESENSE_PORT,
                protocol=RAGConfig.TYPESENSE_PROTOCOL,
                api_key=RAGConfig.TYPESENSE_API_KEY,
                batch_size=RAGConfig.BATCH_SIZE,
            )
            self.storage = VectorStoragePipeline(storage_config)
            log.info("✓ Vector Storage Pipeline initialized (Typesense)")
            
            # Stage 5: LLM
            llm_config = LLMConfig(
                provider="groq",
                api_key=RAGConfig.GROQ_API_KEY,
                model_name=RAGConfig.LLM_MODEL,
                temperature=RAGConfig.LLM_TEMPERATURE,
                max_tokens=RAGConfig.LLM_MAX_TOKENS,
                context_score_threshold=RAGConfig.CONTEXT_THRESHOLD,
                max_context_length=RAGConfig.MAX_CONTEXT_LENGTH,
            )
            self.llm = LLMPipeline(llm_config, self.embedder, self.storage)
            log.info(f"✓ LLM Pipeline initialized ({RAGConfig.LLM_MODEL})")
            
            log.info("All pipelines initialized successfully!")
            
        except Exception as e:
            log.error(f"Pipeline initialization failed: {e}")
            raise
    
    def get_conversation(self, conversation_id: str):
        """Get or create conversation."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                "history": [],
                "llm_instance": None,  # Each conversation has its own LLM instance
            }
        return self.conversations[conversation_id]
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
    
    def health_check(self) -> dict:
        """Check health of all pipelines."""
        return {
            "ingestion": self.ingestion is not None,
            "chunker": self.chunker is not None,
            "embedder": self.embedder is not None and self.embedder.is_ready,
            "storage": self.storage is not None and self.storage.ping(),
            "llm": self.llm is not None,
        }


# Global instance
pipeline_manager = PipelineManager()


def get_pipeline_manager() -> PipelineManager:
    """Dependency injection for FastAPI."""
    return pipeline_manager