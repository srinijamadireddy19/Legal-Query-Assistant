"""
Embedding Pipeline
──────────────────
Main entry point for generating embeddings from chunks.
Handles model loading, fallback, batching, and error recovery.

Usage:
    from embedding_pipeline import EmbeddingPipeline, EmbeddingConfig
    
    config = EmbeddingConfig(
        model_name="BAAI/bge-large-en-v1.5",
        fallback_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    
    pipeline = EmbeddingPipeline(config)
    result = pipeline.embed_chunks(chunks)
"""

import logging
import time
from typing import List, Union, Dict, Any
from pathlib import Path

# Import all embedders to register them
from .embedders import fastembed_embedder, sentence_transformers_embedder  # noqa
from .core.base_embedder import registry, BaseEmbedder
from .core.models import (
    ChunkEmbedding, 
    EmbeddingResult, 
    EmbeddingConfig, 
    EmbeddingStatus
)

log = logging.getLogger(__name__)


class EmbeddingPipeline:
    """
    Main pipeline for embedding generation.
    Supports multiple embedding backends with automatic fallback.
    """
    
    def __init__(self, config: EmbeddingConfig = None):
        """
        Initialize embedding pipeline.
        
        Args:
            config: EmbeddingConfig or None (uses defaults)
        """
        self.config = config or EmbeddingConfig()
        self.config.validate()
        
        self.embedder: BaseEmbedder = None
        self.fallback_embedder: BaseEmbedder = None
        
        # Try loading primary model
        if not self._load_primary_model():
            log.warning(f"Primary model {self.config.model_name} failed to load")
            
            # Try fallback
            if self.config.fallback_model:
                log.info(f"Attempting fallback: {self.config.fallback_model}")
                if not self._load_fallback_model():
                    raise RuntimeError(
                        f"Both primary ({self.config.model_name}) and "
                        f"fallback ({self.config.fallback_model}) models failed to load"
                    )
            else:
                raise RuntimeError(
                    f"Failed to load model {self.config.model_name} and no fallback configured"
                )
    
    def _load_primary_model(self) -> bool:
        """Load the primary embedding model."""
        try:
            embedder_cls = registry.get(self.config.model_name)
            self.embedder = embedder_cls(self.config)
            return self.embedder.load_model()
        except Exception as e:
            log.error(f"Primary model load failed: {e}")
            return False
    
    def _load_fallback_model(self) -> bool:
        """Load the fallback embedding model."""
        try:
            # Create config for fallback model
            fallback_config = EmbeddingConfig(
                model_name=self.config.fallback_model,
                batch_size=self.config.batch_size,
                device=self.config.device,
                max_length=self.config.max_length,
                normalize=self.config.normalize,
            )
            
            embedder_cls = registry.get(fallback_config.model_name)
            self.fallback_embedder = embedder_cls(fallback_config)
            success = self.fallback_embedder.load_model()
            
            if success:
                # Use fallback as primary now
                self.embedder = self.fallback_embedder
                self.config.model_name = self.config.fallback_model
                log.info(f"✓ Switched to fallback model: {self.config.fallback_model}")
            
            return success
            
        except Exception as e:
            log.error(f"Fallback model load failed: {e}")
            return False
    
    # ── Embedding from Chunks (from hierarchical chunker) ─────────────────
    
    def embed_chunks(
        self,
        chunks: List[Any],  # List of Chunk objects from hierarchical chunker
        source_file: str = "",
    ) -> EmbeddingResult:
        """
        Embed chunks from the hierarchical chunking pipeline.
        
        Args:
            chunks: List of Chunk objects (from legal_rag_chunker)
            source_file: Original file path for provenance
            
        Returns:
            EmbeddingResult with all embeddings + stats
        """
        if not chunks:
            return EmbeddingResult(
                embeddings=[],
                status=EmbeddingStatus.SUCCESS,
            )
        
        start_time = time.time()
        
        # Extract text and metadata from chunks
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = self._embed_with_retry(texts)
        
        # Package as ChunkEmbedding objects
        chunk_embeddings: List[ChunkEmbedding] = []
        errors: List[str] = []
        
        for chunk, embedding in zip(chunks, embeddings):
            if embedding:
                chunk_emb = ChunkEmbedding(
                    chunk_id=chunk.chunk_id,
                    text=chunk.content,
                    embedding=embedding,
                    metadata=chunk.metadata,
                    model_name=self.embedder.model_info.name,
                    hierarchy=chunk.hierarchy,
                    source_file=source_file,
                )
                chunk_embeddings.append(chunk_emb)
            else:
                errors.append(f"Failed to embed chunk: {chunk.chunk_id}")
        
        total_time = time.time() - start_time
        
        status = self._determine_status(chunk_embeddings, chunks)
        
        return EmbeddingResult(
            embeddings=chunk_embeddings,
            status=status,
            total_time_sec=total_time,
            avg_time_per_chunk=total_time / len(chunks) if chunks else 0,
            errors=errors,
        )
    
    # ── Embedding from raw texts ──────────────────────────────────────────
    
    def embed_texts(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        """
        Embed raw text strings.
        
        Args:
            texts: List of text strings
            metadata: Optional metadata for each text
            
        Returns:
            EmbeddingResult with all embeddings + stats
        """
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                status=EmbeddingStatus.SUCCESS,
            )
        
        start_time = time.time()
        
        # Generate embeddings
        embeddings = self._embed_with_retry(texts)
        
        # Package as ChunkEmbedding objects
        chunk_embeddings: List[ChunkEmbedding] = []
        errors: List[str] = []
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            if embedding:
                chunk_emb = ChunkEmbedding(
                    chunk_id=f"text_{i}",
                    text=text,
                    embedding=embedding,
                    metadata=metadata[i] if metadata and i < len(metadata) else {},
                    model_name=self.embedder.model_info.name,
                )
                chunk_embeddings.append(chunk_emb)
            else:
                errors.append(f"Failed to embed text #{i}")
        
        total_time = time.time() - start_time
        
        status = self._determine_status(chunk_embeddings, texts)
        
        return EmbeddingResult(
            embeddings=chunk_embeddings,
            status=status,
            total_time_sec=total_time,
            avg_time_per_chunk=total_time / len(texts) if texts else 0,
            errors=errors,
        )
    
    # ── Core embedding with retry logic ───────────────────────────────────
    
    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Embed texts with retry logic."""
        for attempt in range(self.config.max_retries + 1):
            try:
                embeddings = self.embedder.embed(texts)
                return embeddings
            except Exception as e:
                if attempt < self.config.max_retries:
                    log.warning(
                        f"Embedding attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {self.config.retry_delay_sec}s..."
                    )
                    time.sleep(self.config.retry_delay_sec)
                else:
                    log.error(f"All embedding attempts failed: {e}")
                    return [[] for _ in texts]
    
    # ── Utility methods ───────────────────────────────────────────────────
    
    def _determine_status(
        self, 
        chunk_embeddings: List[ChunkEmbedding], 
        original_items: List,
    ) -> EmbeddingStatus:
        """Determine overall status based on success rate."""
        total = len(original_items)
        successful = len([e for e in chunk_embeddings if e.embedding])
        
        if successful == 0:
            return EmbeddingStatus.FAILED
        elif successful < total:
            return EmbeddingStatus.PARTIAL
        else:
            return EmbeddingStatus.SUCCESS
    
    @property
    def model_info(self):
        """Get information about the loaded model."""
        return self.embedder.model_info if self.embedder else None
    
    @property
    def is_ready(self) -> bool:
        """Check if pipeline is ready to embed."""
        return self.embedder is not None and self.embedder.is_loaded()
    
    def __repr__(self) -> str:
        if self.is_ready:
            return f"EmbeddingPipeline({self.model_info.name}, {self.model_info.dimension}d)"
        else:
            return "EmbeddingPipeline(not ready)"