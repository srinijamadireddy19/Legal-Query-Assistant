"""
Paragraph Chunker - Final fallback for unstructured documents
Simple paragraph-based chunking with smart merging
"""

import re
from typing import List, Dict
from ..chunker_factory import BaseChunker, Chunk, DetectionResult


class ParagraphChunker(BaseChunker):
    """
    Fallback chunker for unstructured documents
    Uses paragraph boundaries and smart merging
    """
    
    def _default_config(self) -> Dict:
        return {
            'max_chunk_size': 1000,
            'min_chunk_size': 200,
            'overlap': 100,
            'sentence_overlap': True,  # Overlap at sentence boundaries
            'preserve_sentence_integrity': True,
        }
    
    def chunk(self, text: str, detection_result: DetectionResult) -> List[Chunk]:
        """
        Chunk unstructured document by paragraphs
        
        Strategy:
        1. Split by paragraphs (double newline)
        2. Merge small paragraphs
        3. Split large paragraphs by sentences
        4. Add overlap between chunks
        """
        text = self._clean_text(text)
        
        # Split into paragraphs
        paragraphs = self._extract_paragraphs(text)
        
        # Create chunks from paragraphs
        chunks = self._create_chunks_from_paragraphs(paragraphs)
        
        # Add overlap if configured
        if self.config.get('overlap', 0) > 0:
            chunks = self._add_overlap(chunks)
        
        return chunks
    
    def _extract_paragraphs(self, text: str) -> List[Dict]:
        """
        Extract paragraphs with metadata
        
        Returns list of dicts with:
        - content: paragraph text
        - start: character position
        - end: character position
        - sentences: list of sentences
        """
        paragraphs = []
        
        # Split by double newline or more
        para_texts = re.split(r'\n\s*\n+', text)
        
        current_pos = 0
        for para_text in para_texts:
            para_text = para_text.strip()
            
            if not para_text:
                current_pos += len(para_text) + 2
                continue
            
            # Find actual position in original text
            start_pos = text.find(para_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            
            end_pos = start_pos + len(para_text)
            
            # Extract sentences
            sentences = self._split_into_sentences(para_text)
            
            paragraphs.append({
                'content': para_text,
                'start': start_pos,
                'end': end_pos,
                'sentences': sentences
            })
            
            current_pos = end_pos + 2
        
        return paragraphs
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple rules
        """
        # Simple sentence splitter - can be replaced with spacy/nltk
        sentences = []
        
        # Pattern for sentence endings
        pattern = r'([.!?]+["\'»]?\s+)(?=[A-Z])'
        
        parts = re.split(pattern, text)
        
        current_sentence = ""
        for part in parts:
            current_sentence += part
            if re.match(pattern, part):
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence:
            sentences.append(current_sentence.strip())
        
        return [s for s in sentences if s]
    
    def _create_chunks_from_paragraphs(self, paragraphs: List[Dict]) -> List[Chunk]:
        """
        Create chunks from paragraphs with smart merging
        """
        if not paragraphs:
            return []
        
        chunks = []
        current_chunk_paras = []
        current_size = 0
        chunk_id = 0
        
        for para in paragraphs:
            para_size = len(para['content'])
            
            # Check if adding this paragraph would exceed max size
            if current_size + para_size > self.config['max_chunk_size'] and current_chunk_paras:
                # Save current chunk
                chunk = self._create_chunk_from_paragraphs(
                    current_chunk_paras,
                    chunk_id
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_paras = [para]
                current_size = para_size
                chunk_id += 1
            else:
                # Add to current chunk
                current_chunk_paras.append(para)
                current_size += para_size
        
        # Add final chunk
        if current_chunk_paras:
            chunk = self._create_chunk_from_paragraphs(
                current_chunk_paras,
                chunk_id
            )
            chunks.append(chunk)
        
        # Split any chunks that are still too large
        final_chunks = []
        for chunk in chunks:
            if len(chunk.content) > self.config['max_chunk_size']:
                split_chunks = self._split_large_chunk(chunk)
                final_chunks.extend(split_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _create_chunk_from_paragraphs(
        self,
        paragraphs: List[Dict],
        chunk_id: int
    ) -> Chunk:
        """Create a chunk from list of paragraphs"""
        content = '\n\n'.join(p['content'] for p in paragraphs)
        start_pos = paragraphs[0]['start']
        end_pos = paragraphs[-1]['end']
        
        return Chunk(
            content=content,
            chunk_id=f"para_{chunk_id}",
            metadata={
                'type': 'paragraph',
                'paragraph_count': len(paragraphs),
                'is_fallback': True
            },
            hierarchy=[],
            start_pos=start_pos,
            end_pos=end_pos
        )
    
    def _split_large_chunk(self, chunk: Chunk) -> List[Chunk]:
        """
        Split a chunk that's too large
        Uses sentence boundaries if possible
        """
        if not self.config.get('preserve_sentence_integrity'):
            # Simple character split
            return self._simple_split(chunk)
        
        # Split by sentences
        sentences = self._split_into_sentences(chunk.content)
        
        sub_chunks = []
        current_text = ""
        part_num = 0
        
        for sentence in sentences:
            if len(current_text) + len(sentence) > self.config['max_chunk_size'] and current_text:
                # Save current
                sub_chunks.append(Chunk(
                    content=current_text.strip(),
                    chunk_id=f"{chunk.chunk_id}_part{part_num}",
                    metadata={
                        **chunk.metadata,
                        'is_partial': True,
                        'part_number': part_num
                    },
                    hierarchy=chunk.hierarchy,
                    start_pos=chunk.start_pos,
                    end_pos=chunk.start_pos + len(current_text)
                ))
                
                current_text = sentence
                part_num += 1
            else:
                current_text = current_text + " " + sentence if current_text else sentence
        
        # Add final part
        if current_text:
            sub_chunks.append(Chunk(
                content=current_text.strip(),
                chunk_id=f"{chunk.chunk_id}_part{part_num}",
                metadata={
                    **chunk.metadata,
                    'is_partial': part_num > 0,
                    'part_number': part_num
                },
                hierarchy=chunk.hierarchy,
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos
            ))
        
        return sub_chunks
    
    def _simple_split(self, chunk: Chunk) -> List[Chunk]:
        """Simple character-based split"""
        max_size = self.config['max_chunk_size']
        content = chunk.content
        
        sub_chunks = []
        start = 0
        part_num = 0
        
        while start < len(content):
            end = start + max_size
            sub_content = content[start:end]
            
            sub_chunks.append(Chunk(
                content=sub_content,
                chunk_id=f"{chunk.chunk_id}_part{part_num}",
                metadata={
                    **chunk.metadata,
                    'is_partial': True,
                    'part_number': part_num
                },
                hierarchy=chunk.hierarchy,
                start_pos=chunk.start_pos + start,
                end_pos=chunk.start_pos + end
            ))
            
            start = end
            part_num += 1
        
        return sub_chunks
    
    def _add_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Add overlap between consecutive chunks
        """
        if len(chunks) <= 1:
            return chunks
        
        overlap_size = self.config['overlap']
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.content
            
            # Add overlap from previous chunk
            if i > 0 and self.config.get('sentence_overlap'):
                prev_chunk = chunks[i-1]
                overlap_text = self._get_last_sentences(prev_chunk.content, overlap_size)
                content = f"[...{overlap_text}]\n\n{content}"
            
            # Create new chunk with overlap
            new_chunk = Chunk(
                content=content,
                chunk_id=chunk.chunk_id,
                metadata={
                    **chunk.metadata,
                    'has_overlap': i > 0
                },
                hierarchy=chunk.hierarchy,
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos
            )
            
            overlapped_chunks.append(new_chunk)
        
        return overlapped_chunks
    
    def _get_last_sentences(self, text: str, max_chars: int) -> str:
        """Get last few sentences up to max_chars"""
        sentences = self._split_into_sentences(text)
        
        # Work backwards to get sentences that fit
        result = []
        current_size = 0
        
        for sentence in reversed(sentences):
            if current_size + len(sentence) > max_chars:
                break
            result.insert(0, sentence)
            current_size += len(sentence)
        
        return ' '.join(result)
