"""
Chunker Factory - Routes to appropriate chunking strategy based on document type
"""

from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .document_structure_detector import DocumentType, DetectionResult


@dataclass
class Chunk:
    """Represents a single chunk with metadata"""
    content: str
    chunk_id: str
    metadata: Dict
    hierarchy: List[str]  # Breadcrumb trail: ["Section 1", "Subsection 1.1"]
    start_pos: int
    end_pos: int


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
    
    @abstractmethod
    def _default_config(self) -> Dict:
        """Return default configuration for this chunker"""
        pass
    
    @abstractmethod
    def chunk(self, text: str, detection_result: DetectionResult) -> List[Chunk]:
        """
        Chunk the document according to strategy
        
        Args:
            text: Document text
            detection_result: Result from structure detection
            
        Returns:
            List of chunks with metadata and hierarchy
        """
        pass
    
    def _clean_text(self, text: str) -> str:
        """Common text cleaning operations"""
        # Remove excessive whitespace
        text = '\n'.join(line.strip() for line in text.split('\n'))
        # Remove multiple blank lines
        text = '\n'.join(line for line in text.split('\n') if line or not text)
        return text.strip()
    
    def _merge_small_chunks(
        self, 
        chunks: List[Chunk], 
        min_size: int = 100
    ) -> List[Chunk]:
        """Merge chunks that are too small"""
        if not chunks:
            return chunks
        
        merged = []
        current = chunks[0]
        
        for next_chunk in chunks[1:]:
            if len(current.content) < min_size and len(next_chunk.content) < min_size:
                # Merge with next
                current = Chunk(
                    content=current.content + "\n\n" + next_chunk.content,
                    chunk_id=current.chunk_id,
                    metadata={**current.metadata, 'merged': True},
                    hierarchy=current.hierarchy,
                    start_pos=current.start_pos,
                    end_pos=next_chunk.end_pos
                )
            else:
                merged.append(current)
                current = next_chunk
        
        merged.append(current)
        return merged


class ChunkerFactory:
    """Factory to create appropriate chunker based on document type"""
    
    def __init__(self):
        self._chunkers = {
            DocumentType.NUMBERED_SECTIONS: NumberedSectionChunker,
            DocumentType.STATUTE: StatuteChunker,
            DocumentType.POLICY_OR_TERMS: PolicyChunker,
            DocumentType.LAYOUT_BASED: LayoutBasedChunker,
            DocumentType.UNSTRUCTURED: ParagraphChunker,
        }
    
    def create_chunker(
        self, 
        doc_type: DocumentType,
        config: Optional[Dict] = None
    ) -> BaseChunker:
        """
        Create appropriate chunker for document type
        
        Args:
            doc_type: Detected document type
            config: Optional configuration
            
        Returns:
            Instantiated chunker
        """
        chunker_class = self._chunkers.get(doc_type)
        if not chunker_class:
            raise ValueError(f"No chunker available for document type: {doc_type}")
        
        return chunker_class(config)
    
    def chunk_document(
        self,
        text: str,
        detection_result: DetectionResult,
        config: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Convenience method to detect and chunk in one call
        
        Args:
            text: Document text
            detection_result: Result from structure detection
            config: Optional configuration
            
        Returns:
            List of chunks
        """
        chunker = self.create_chunker(detection_result.doc_type, config)
        return chunker.chunk(text, detection_result)


# Import chunker implementations
from .chunkers.numbered_section_chunker import NumberedSectionChunker
from .chunkers.statute_chunker import StatuteChunker
from .chunkers.policy_chunker import PolicyChunker
from .chunkers.layout_based_chunker import LayoutBasedChunker
from .chunkers.paragraph_chunker import ParagraphChunker
