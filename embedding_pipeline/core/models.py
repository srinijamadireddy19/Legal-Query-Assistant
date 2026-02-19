"""
Core data models for the embedding pipeline.
Represents chunks with their embeddings and metadata.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import hashlib


class EmbeddingStatus(Enum):
    """Status of embedding generation."""
    SUCCESS = "success"
    FAILED  = "failed"
    PARTIAL = "partial"


@dataclass
class ChunkEmbedding:
    """
    A single chunk with its embedding vector.
    This is what gets stored in vector DBs.
    """
    chunk_id:       str                    # Unique identifier
    text:           str                    # Original chunk text
    embedding:      List[float]            # Dense vector (384/768/1024 dims)
    
    # Metadata for retrieval + filtering
    metadata:       Dict[str, Any] = field(default_factory=dict)
    
    # Model info for tracking
    model_name:     str = ""
    embedding_dim:  int = 0
    
    # Hierarchy from chunker (if available)
    hierarchy:      List[str] = field(default_factory=list)
    
    # Provenance
    source_file:    str = ""
    page_numbers:   List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if self.embedding:
            self.embedding_dim = len(self.embedding)
        
        # Generate chunk_id from content hash if not provided
        if not self.chunk_id:
            self.chunk_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate deterministic ID from text content."""
        content_hash = hashlib.sha256(self.text.encode('utf-8')).hexdigest()[:16]
        return f"chunk_{content_hash}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "hierarchy": self.hierarchy,
            "source_file": self.source_file,
            "page_numbers": self.page_numbers,
        }


@dataclass
class EmbeddingResult:
    """
    Result of embedding a batch of chunks.
    Contains all embeddings + summary statistics.
    """
    embeddings:     List[ChunkEmbedding]
    status:         EmbeddingStatus
    
    # Stats
    total_chunks:   int = 0
    successful:     int = 0
    failed:         int = 0
    
    # Model used
    model_name:     str = ""
    embedding_dim:  int = 0
    
    # Performance
    total_time_sec: float = 0.0
    avg_time_per_chunk: float = 0.0
    
    # Errors
    errors:         List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.total_chunks = len(self.embeddings)
        self.successful = len([e for e in self.embeddings if e.embedding])
        self.failed = self.total_chunks - self.successful
        
        if self.embeddings and self.embeddings[0].embedding:
            self.embedding_dim = len(self.embeddings[0].embedding)
            self.model_name = self.embeddings[0].model_name
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Embedding Result",
            f"  Model         : {self.model_name}",
            f"  Dimension     : {self.embedding_dim}",
            f"  Status        : {self.status.value}",
            f"  Total chunks  : {self.total_chunks}",
            f"  Successful    : {self.successful}",
            f"  Failed        : {self.failed}",
            f"  Total time    : {self.total_time_sec:.2f}s",
        ]
        if self.successful > 0:
            lines.append(f"  Avg time/chunk: {self.avg_time_per_chunk*1000:.1f}ms")
        if self.errors:
            lines.append(f"  Errors        : {len(self.errors)}")
        return "\n".join(lines)


@dataclass  
class EmbeddingConfig:
    """
    Configuration for embedding models.
    Provides sensible defaults, allows overrides.
    """
    # Primary model
    model_name:      str = "BAAI/bge-large-en-v1.5"
    
    # Fallback model (used if primary fails to load)
    fallback_model:  Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Batch processing
    batch_size:      int = 32
    
    # Model loading
    cache_dir:       Optional[str] = None
    device:          str = "cpu"  # "cpu" or "cuda"
    
    # Text processing
    max_length:      int = 512    # Max tokens per chunk
    normalize:       bool = True  # L2 normalize embeddings
    
    # Performance
    show_progress:   bool = True
    num_workers:     int = 4      # For parallel processing
    
    # Retry logic
    max_retries:     int = 2
    retry_delay_sec: float = 1.0
    
    def validate(self):
        """Validate configuration."""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.max_length > 0, "max_length must be positive"
        assert self.device in ("cpu", "cuda"), "device must be 'cpu' or 'cuda'"


@dataclass
class ModelInfo:
    """Information about an embedding model."""
    name:           str
    dimension:      int
    max_seq_length: int
    description:    str
    size_mb:        float = 0.0
    
    def __str__(self) -> str:
        return (
            f"{self.name} "
            f"(dim={self.dimension}, max_len={self.max_seq_length})"
        )