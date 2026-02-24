"""
Hierarchical Chunking Pipeline - Main orchestrator
Combines detection and chunking into a single pipeline
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from .core.document_structure_detector import DocumentStructureDetector, DetectionResult
from .core.chunker_factory import ChunkerFactory, Chunk


@dataclass
class ChunkingResult:
    """Complete result from chunking pipeline"""
    chunks: List[Chunk]
    detection_result: DetectionResult
    statistics: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'chunks': [
                {
                    'content': c.content,
                    'chunk_id': c.chunk_id,
                    'metadata': c.metadata,
                    'hierarchy': c.hierarchy,
                    'start_pos': c.start_pos,
                    'end_pos': c.end_pos
                }
                for c in self.chunks
            ],
            'detection': {
                'doc_type': self.detection_result.doc_type.value,
                'confidence': self.detection_result.confidence,
                'metadata': self.detection_result.metadata,
                'structure_hints': self.detection_result.structure_hints
            },
            'statistics': self.statistics
        }


class HierarchicalChunkingPipeline:
    """
    Main pipeline that orchestrates document detection and chunking
    
    Usage:
        pipeline = HierarchicalChunkingPipeline()
        result = pipeline.process(document_text)
        chunks = result.chunks
    """
    
    def __init__(
        self,
        detector_config: Optional[Dict] = None,
        chunker_config: Optional[Dict] = None
    ):
        """
        Initialize pipeline with optional configurations
        
        Args:
            detector_config: Configuration for document structure detector
            chunker_config: Configuration for chunkers
        """
        self.detector = DocumentStructureDetector(detector_config)
        self.factory = ChunkerFactory()
        self.chunker_config = chunker_config or {}
    
    def process(
        self,
        text: str,
        layout_info: Optional[Dict] = None,
        custom_config: Optional[Dict] = None
    ) -> ChunkingResult:
        """
        Process a document through the complete pipeline
        
        Args:
            text: Document text
            layout_info: Optional layout information from PDF parser
            custom_config: Optional override configuration for this document
            
        Returns:
            ChunkingResult with chunks and metadata
        """
        # Stage 1: Detect document structure
        detection_result = self.detector.detect(text, layout_info)
        
        # Merge layout_info into detection result if provided
        if layout_info:
            detection_result.metadata['layout_info'] = layout_info
        
        # Stage 2: Get appropriate chunker configuration
        config = custom_config or self.chunker_config
        
        # Stage 3: Chunk the document
        chunks = self.factory.chunk_document(text, detection_result, config)
        
        # Stage 4: Calculate statistics
        statistics = self._calculate_statistics(chunks, text)
        
        return ChunkingResult(
            chunks=chunks,
            detection_result=detection_result,
            statistics=statistics
        )
    
    def process_batch(
        self,
        documents: List[Dict],
        show_progress: bool = False
    ) -> List[ChunkingResult]:
        """
        Process multiple documents
        
        Args:
            documents: List of dicts with 'text' and optional 'layout_info'
            show_progress: Show progress bar (requires tqdm)
            
        Returns:
            List of ChunkingResults
        """
        results = []
        
        iterator = documents
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(documents, desc="Chunking documents")
            except ImportError:
                pass
        
        for doc in iterator:
            text = doc.get('text', '')
            layout_info = doc.get('layout_info')
            doc_id = doc.get('id', 'unknown')
            
            try:
                result = self.process(text, layout_info)
                results.append(result)
            except Exception as e:
                print(f"Error processing document {doc_id}: {e}")
                # Continue with other documents
        
        return results
    
    def _calculate_statistics(self, chunks: List[Chunk], original_text: str) -> Dict:
        """Calculate statistics about the chunking result"""
        if not chunks:
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'original_size': len(original_text)
            }
        
        chunk_sizes = [len(c.content) for c in chunks]
        
        # Count chunks by type
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk.metadata.get('type', 'unknown')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        # Hierarchy depth distribution
        hierarchy_depths = [len(c.hierarchy) for c in chunks]
        max_depth = max(hierarchy_depths) if hierarchy_depths else 0
        avg_depth = sum(hierarchy_depths) / len(hierarchy_depths) if hierarchy_depths else 0
        
        return {
            'total_chunks': len(chunks),
            'total_characters': sum(chunk_sizes),
            'avg_chunk_size': sum(chunk_sizes) / len(chunks),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'chunk_types': chunk_types,
            'max_hierarchy_depth': max_depth,
            'avg_hierarchy_depth': avg_depth,
            'original_size': len(original_text),
            'compression_ratio': len(original_text) / sum(chunk_sizes) if sum(chunk_sizes) > 0 else 0
        }
    
    def get_chunk_by_id(self, chunks: List[Chunk], chunk_id: str) -> Optional[Chunk]:
        """Helper method to retrieve a chunk by ID"""
        for chunk in chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def get_chunks_by_hierarchy(
        self,
        chunks: List[Chunk],
        hierarchy_path: List[str]
    ) -> List[Chunk]:
        """
        Get all chunks under a specific hierarchy path
        
        Args:
            chunks: List of chunks to search
            hierarchy_path: Path to match (e.g., ["Section 1", "Subsection 1.1"])
            
        Returns:
            List of matching chunks
        """
        matching_chunks = []
        
        for chunk in chunks:
            # Check if chunk hierarchy starts with the search path
            if len(chunk.hierarchy) >= len(hierarchy_path):
                if chunk.hierarchy[:len(hierarchy_path)] == hierarchy_path:
                    matching_chunks.append(chunk)
        
        return matching_chunks
    
    def export_chunks(
        self,
        chunks: List[Chunk],
        format: str = 'json',
        include_metadata: bool = True
    ) -> str:
        """
        Export chunks to various formats
        
        Args:
            chunks: Chunks to export
            format: 'json', 'csv', or 'markdown'
            include_metadata: Whether to include metadata
            
        Returns:
            Formatted string
        """
        if format == 'json':
            return self._export_json(chunks, include_metadata)
        elif format == 'csv':
            return self._export_csv(chunks, include_metadata)
        elif format == 'markdown':
            return self._export_markdown(chunks, include_metadata)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_json(self, chunks: List[Chunk], include_metadata: bool) -> str:
        """Export to JSON format"""
        import json
        
        data = []
        for chunk in chunks:
            chunk_data = {
                'id': chunk.chunk_id,
                'content': chunk.content,
                'hierarchy': chunk.hierarchy,
            }
            
            if include_metadata:
                chunk_data.update({
                    'metadata': chunk.metadata,
                    'start_pos': chunk.start_pos,
                    'end_pos': chunk.end_pos
                })
            
            data.append(chunk_data)
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def _export_csv(self, chunks: List[Chunk], include_metadata: bool) -> str:
        """Export to CSV format"""
        import csv
        from io import StringIO
        
        output = StringIO()
        
        fieldnames = ['chunk_id', 'content', 'hierarchy']
        if include_metadata:
            fieldnames.extend(['metadata', 'start_pos', 'end_pos'])
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for chunk in chunks:
            row = {
                'chunk_id': chunk.chunk_id,
                'content': chunk.content.replace('\n', ' '),
                'hierarchy': ' > '.join(chunk.hierarchy)
            }
            
            if include_metadata:
                row.update({
                    'metadata': str(chunk.metadata),
                    'start_pos': chunk.start_pos,
                    'end_pos': chunk.end_pos
                })
            
            writer.writerow(row)
        
        return output.getvalue()
    
    def _export_markdown(self, chunks: List[Chunk], include_metadata: bool) -> str:
        """Export to Markdown format"""
        lines = ["# Document Chunks\n"]
        
        for i, chunk in enumerate(chunks, 1):
            # Hierarchy as heading
            hierarchy_level = min(len(chunk.hierarchy), 6)
            if chunk.hierarchy:
                hierarchy_text = ' > '.join(chunk.hierarchy)
                lines.append(f"{'#' * hierarchy_level} {hierarchy_text}\n")
            else:
                lines.append(f"## Chunk {i}\n")
            
            # Metadata
            if include_metadata:
                lines.append(f"**ID:** `{chunk.chunk_id}`  ")
                lines.append(f"**Type:** {chunk.metadata.get('type', 'unknown')}  ")
                lines.append(f"**Position:** {chunk.start_pos}-{chunk.end_pos}\n")
            
            # Content
            lines.append(f"{chunk.content}\n")
            lines.append("\n---\n")
        
        return '\n'.join(lines)