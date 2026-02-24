"""
Legal RAG Chunker - Hierarchical Document Chunking System

A modular system for intelligently chunking legal documents based on their structure.
Supports numbered sections, statutes, policies, and unstructured documents.

Usage:
    from legal_rag_chunker import HierarchicalChunkingPipeline
    
    pipeline = HierarchicalChunkingPipeline()
    result = pipeline.process(document_text)
    
    for chunk in result.chunks:
        print(f"{chunk.chunk_id}: {chunk.hierarchy}")
        print(chunk.content)
"""

from .pipeline import HierarchicalChunkingPipeline, ChunkingResult
from .core import (
    DocumentStructureDetector,
    DocumentType,
    DetectionResult,
    ChunkerFactory,
    Chunk
)

__version__ = "1.0.0"

__all__ = [
    'HierarchicalChunkingPipeline',
    'ChunkingResult',
    'DocumentStructureDetector',
    'DocumentType',
    'DetectionResult',
    'ChunkerFactory',
    'Chunk'
]