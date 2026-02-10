"""
Legal RAG Chunker - Core Package
Hierarchical document chunking for legal documents
"""

from .document_structure_detector import (
    DocumentStructureDetector,
    DocumentType,
    DetectionResult
)
from .chunker_factory import (
    ChunkerFactory,
    BaseChunker,
    Chunk
)

__all__ = [
    'DocumentStructureDetector',
    'DocumentType',
    'DetectionResult',
    'ChunkerFactory',
    'BaseChunker',
    'Chunk'
]
