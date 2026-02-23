"""
Core data models for vector storage pipeline.
Defines schemas, indexing results, and storage configurations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class IndexingStatus(Enum):
    """Status of indexing operation."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED  = "failed"


@dataclass
class CollectionSchema:
    """
    Schema definition for a vector collection.
    Typesense-compatible schema with vector search support.
    """
    name: str
    embedding_dim: int
    
    # Optional metadata fields
    fields: List[Dict[str, Any]] = field(default_factory=list)
    
    # Default token separators for text search
    token_separators: List[str] = field(default_factory=lambda: [" ", ",", ".", ":", ";"])
    
    # Version tracking
    schema_version: str = "1.0"
    
    def to_typesense_schema(self) -> Dict[str, Any]:
        """
        Convert to Typesense schema format.
        
        Typesense schema structure:
        {
            "name": "collection_name",
            "fields": [
                {"name": "id", "type": "string"},
                {"name": "embedding", "type": "float[]", "num_dim": 1024},
                {"name": "text", "type": "string"},
                {"name": "metadata", "type": "object"},
                ...
            ],
            "token_separators": [" ", ",", ...],
            "enable_nested_fields": True
        }
        """
        base_fields = [
            {"name": "id", "type": "string"},
            {"name": "embedding", "type": "float[]", "num_dim": self.embedding_dim},
            {"name": "text", "type": "string"},
            {"name": "chunk_id", "type": "string"},
            {"name": "source_file", "type": "string", "optional": True},
            {"name": "hierarchy", "type": "string[]", "optional": True},
            {"name": "page_numbers", "type": "int32[]", "optional": True},
            {"name": "embedding_model", "type": "string", "optional": True},
            {"name": "embedding_model_version", "type": "string", "optional": True},
            {"name": "indexed_at", "type": "int64"},  # Unix timestamp
            {"name": "metadata", "type": "object", "optional": True},
        ]
        
        # Add custom fields
        all_fields = base_fields + self.fields
        
        return {
            "name": self.name,
            "fields": all_fields,
            "token_separators": self.token_separators,
            "enable_nested_fields": True,
        }
    
    def validate(self) -> List[str]:
        """Validate schema definition. Returns list of errors."""
        errors = []
        
        if not self.name:
            errors.append("Collection name is required")
        
        if self.embedding_dim <= 0:
            errors.append(f"Invalid embedding_dim: {self.embedding_dim}")
        
        # Check for duplicate field names
        field_names = set()
        for f in self.fields:
            name = f.get("name")
            if not name:
                errors.append("Field missing 'name' attribute")
            elif name in field_names:
                errors.append(f"Duplicate field name: {name}")
            field_names.add(name)
        
        return errors


@dataclass
class IndexingResult:
    """
    Result of a batch indexing operation.
    Tracks success/failure stats and provides detailed reporting.
    """
    collection_name: str
    status: IndexingStatus
    
    # Statistics
    total_documents: int = 0
    successful: int = 0
    failed: int = 0
    duplicates_skipped: int = 0
    
    # Performance
    total_time_sec: float = 0.0
    avg_time_per_doc: float = 0.0
    docs_per_second: float = 0.0
    
    # Details
    failed_ids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    embedding_dim: int = 0
    embedding_model: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.total_documents > 0:
            self.avg_time_per_doc = self.total_time_sec / self.total_documents
            if self.total_time_sec > 0:
                self.docs_per_second = self.total_documents / self.total_time_sec
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Indexing Result: {self.collection_name}",
            f"  Status        : {self.status.value}",
            f"  Total docs    : {self.total_documents}",
            f"  Successful    : {self.successful}",
            f"  Failed        : {self.failed}",
        ]
        
        if self.duplicates_skipped > 0:
            lines.append(f"  Duplicates    : {self.duplicates_skipped} (skipped)")
        
        lines.extend([
            f"  Total time    : {self.total_time_sec:.2f}s",
            f"  Speed         : {self.docs_per_second:.1f} docs/sec",
        ])
        
        if self.embedding_model:
            lines.append(f"  Model         : {self.embedding_model} ({self.embedding_dim}d)")
        
        if self.errors:
            lines.append(f"  Errors        : {len(self.errors)}")
        
        if self.warnings:
            lines.append(f"  Warnings      : {len(self.warnings)}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "collection_name": self.collection_name,
            "status": self.status.value,
            "total_documents": self.total_documents,
            "successful": self.successful,
            "failed": self.failed,
            "duplicates_skipped": self.duplicates_skipped,
            "total_time_sec": self.total_time_sec,
            "docs_per_second": self.docs_per_second,
            "failed_ids": self.failed_ids,
            "errors": self.errors,
            "warnings": self.warnings,
            "embedding_dim": self.embedding_dim,
            "embedding_model": self.embedding_model,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class StorageConfig:
    """
    Configuration for vector storage backend.
    """
    # Typesense connection
    host: str = "localhost"
    port: int = 8108
    protocol: str = "http"
    api_key: str = ""
    
    # Connection settings
    connection_timeout_seconds: int = 10
    num_retries: int = 3
    
    # Batch indexing
    batch_size: int = 100
    
    # Validation
    validate_embeddings: bool = True
    check_duplicates: bool = True
    
    # Error handling
    skip_on_error: bool = True  # Continue indexing on individual doc errors
    max_errors: int = 100       # Stop after this many errors
    
    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []
        
        if not self.host:
            errors.append("host is required")
        
        if self.port <= 0 or self.port > 65535:
            errors.append(f"Invalid port: {self.port}")
        
        if self.protocol not in ("http", "https"):
            errors.append(f"Invalid protocol: {self.protocol}. Must be http or https")
        
        if not self.api_key:
            errors.append("api_key is required")
        
        if self.batch_size <= 0:
            errors.append(f"Invalid batch_size: {self.batch_size}")
        
        return errors
    
    def get_connection_string(self) -> str:
        """Get connection string for display (without API key)."""
        return f"{self.protocol}://{self.host}:{self.port}"


@dataclass
class CollectionInfo:
    """Information about an existing collection."""
    name: str
    num_documents: int
    fields: List[Dict[str, Any]]
    created_at: Optional[int] = None
    
    def get_embedding_dim(self) -> Optional[int]:
        """Extract embedding dimension from schema."""
        for field in self.fields:
            if field.get("name") == "embedding" and field.get("type") == "float[]":
                return field.get("num_dim")
        return None
    
    def __str__(self) -> str:
        emb_dim = self.get_embedding_dim()
        return (
            f"Collection: {self.name} "
            f"({self.num_documents} docs, {emb_dim}d embeddings)"
        )
