"""
Layout-Based Chunker - Uses PDF layout information (fonts, formatting)
Relies on parsed layout data from PDF extraction
"""

import re
from typing import List, Dict, Optional
from ..chunker_factory import BaseChunker, Chunk, DetectionResult


class LayoutBasedChunker(BaseChunker):
    """
    Chunks documents using layout information from PDF parser
    Uses font size, bold, centering to identify structure
    """
    
    def _default_config(self) -> Dict:
        return {
            'max_chunk_size': 1500,
            'min_chunk_size': 100,
            'heading_min_font_ratio': 1.2,  # Heading font should be 20% larger
            'use_bold_as_heading': True,
            'use_centered_as_heading': True,
        }
    
    def chunk(self, text: str, detection_result: DetectionResult) -> List[Chunk]:
        """
        Chunk document using layout-based heading detection
        
        Note: Requires layout_info to be passed in detection_result.metadata
        """
        text = self._clean_text(text)
        
        # Get layout info from detection result
        layout_info = detection_result.metadata.get('layout_info', {})
        
        if not layout_info:
            # Fallback to simple paragraph chunking
            return self._fallback_chunk(text)
        
        # Extract headings from layout
        headings = self._extract_headings_from_layout(layout_info, text)
        
        # Create sections based on headings
        sections = self._create_sections(text, headings)
        
        # Generate chunks from sections
        chunks = []
        for i, section in enumerate(sections):
            section_chunks = self._chunk_section(section, i)
            chunks.extend(section_chunks)
        
        # Post-process
        if self.config.get('min_chunk_size'):
            chunks = self._merge_small_chunks(chunks, self.config['min_chunk_size'])
        
        return chunks
    
    def _extract_headings_from_layout(
        self, 
        layout_info: Dict, 
        text: str
    ) -> List[Dict]:
        """
        Extract headings from layout information
        
        Layout info format expected:
        {
            'blocks': [
                {
                    'text': 'Block text',
                    'font_size': 12.0,
                    'is_bold': False,
                    'is_centered': False,
                    'position': {'x': 0, 'y': 100}
                }
            ],
            'base_font_size': 10.0
        }
        """
        headings = []
        blocks = layout_info.get('blocks', [])
        base_font_size = layout_info.get('base_font_size', 10.0)
        
        heading_threshold = base_font_size * self.config['heading_min_font_ratio']
        
        for idx, block in enumerate(blocks):
            block_text = block.get('text', '').strip()
            
            if not block_text or len(block_text) > 200:  # Too long for heading
                continue
            
            is_heading = False
            heading_level = 3  # Default level
            
            # Check font size
            font_size = block.get('font_size', base_font_size)
            if font_size >= heading_threshold:
                is_heading = True
                # Level based on font size
                if font_size >= base_font_size * 1.5:
                    heading_level = 1
                elif font_size >= base_font_size * 1.3:
                    heading_level = 2
            
            # Check bold
            if self.config.get('use_bold_as_heading') and block.get('is_bold'):
                is_heading = True
            
            # Check centered
            if self.config.get('use_centered_as_heading') and block.get('is_centered'):
                is_heading = True
            
            if is_heading:
                # Find position in text
                position = self._find_text_position(block_text, text)
                
                if position is not None:
                    headings.append({
                        'text': block_text,
                        'position': position,
                        'level': heading_level,
                        'font_size': font_size,
                        'is_bold': block.get('is_bold', False),
                        'is_centered': block.get('is_centered', False)
                    })
        
        # Sort by position
        headings.sort(key=lambda x: x['position'])
        
        return headings
    
    def _find_text_position(self, search_text: str, full_text: str) -> Optional[int]:
        """Find the position of text in document"""
        # Clean both texts for comparison
        clean_search = re.sub(r'\s+', ' ', search_text.strip())
        clean_full = re.sub(r'\s+', ' ', full_text)
        
        pos = clean_full.find(clean_search)
        return pos if pos != -1 else None
    
    def _create_sections(self, text: str, headings: List[Dict]) -> List[Dict]:
        """
        Create sections from text based on headings
        
        Each section has:
        - heading (optional)
        - content
        - level
        - hierarchy
        """
        if not headings:
            return [{
                'heading': None,
                'content': text,
                'level': 0,
                'hierarchy': [],
                'start': 0,
                'end': len(text)
            }]
        
        sections = []
        hierarchy_stack = []
        
        for i, heading in enumerate(headings):
            # Extract section content
            start = heading['position']
            end = headings[i+1]['position'] if i+1 < len(headings) else len(text)
            
            section_text = text[start:end]
            
            # Remove heading from content
            content = section_text[len(heading['text']):].strip()
            
            # Update hierarchy stack based on level
            level = heading['level']
            
            # Pop higher or equal levels from stack
            while hierarchy_stack and hierarchy_stack[-1]['level'] >= level:
                hierarchy_stack.pop()
            
            # Build hierarchy breadcrumb
            hierarchy = [h['text'] for h in hierarchy_stack] + [heading['text']]
            
            # Add to stack
            hierarchy_stack.append({
                'text': heading['text'],
                'level': level
            })
            
            sections.append({
                'heading': heading['text'],
                'content': content,
                'level': level,
                'hierarchy': hierarchy,
                'start': start,
                'end': end
            })
        
        return sections
    
    def _chunk_section(self, section: Dict, section_index: int) -> List[Chunk]:
        """Create chunks from a section"""
        chunks = []
        
        heading = section.get('heading')
        content = section['content']
        hierarchy = section['hierarchy']
        
        # If content is small enough, single chunk
        if len(content) <= self.config['max_chunk_size']:
            chunk = Chunk(
                content=content,
                chunk_id=f"section_{section_index}",
                metadata={
                    'section_index': section_index,
                    'heading': heading,
                    'level': section['level'],
                    'type': 'layout_section'
                },
                hierarchy=hierarchy,
                start_pos=section['start'],
                end_pos=section['end']
            )
            chunks.append(chunk)
        else:
            # Split into multiple chunks
            sub_chunks = self._split_large_section(
                content,
                heading,
                hierarchy,
                section_index,
                section['start']
            )
            chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_large_section(
        self,
        content: str,
        heading: str,
        hierarchy: List[str],
        section_index: int,
        start_pos: int
    ) -> List[Chunk]:
        """Split large section into multiple chunks"""
        chunks = []
        max_size = self.config['max_chunk_size']
        
        # Try to split by paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        current_chunk = ""
        chunk_num = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > max_size and current_chunk:
                # Save current chunk
                full_content = f"{heading}\n\n{current_chunk}" if heading else current_chunk
                
                chunks.append(Chunk(
                    content=full_content.strip(),
                    chunk_id=f"section_{section_index}_part_{chunk_num}",
                    metadata={
                        'section_index': section_index,
                        'heading': heading,
                        'is_partial': True,
                        'part_number': chunk_num,
                        'type': 'layout_section'
                    },
                    hierarchy=hierarchy,
                    start_pos=start_pos,
                    end_pos=start_pos + len(current_chunk)
                ))
                
                current_chunk = para
                chunk_num += 1
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        
        # Add final chunk
        if current_chunk:
            full_content = f"{heading}\n\n{current_chunk}" if heading else current_chunk
            
            chunks.append(Chunk(
                content=full_content.strip(),
                chunk_id=f"section_{section_index}_part_{chunk_num}",
                metadata={
                    'section_index': section_index,
                    'heading': heading,
                    'is_partial': chunk_num > 0,
                    'part_number': chunk_num,
                    'type': 'layout_section'
                },
                hierarchy=hierarchy,
                start_pos=start_pos,
                end_pos=start_pos + len(content)
            ))
        
        return chunks
    
    def _fallback_chunk(self, text: str) -> List[Chunk]:
        """Fallback to simple chunking when no layout info available"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > self.config['max_chunk_size'] and current_chunk:
                chunks.append(Chunk(
                    content=current_chunk.strip(),
                    chunk_id=f"fallback_{chunk_id}",
                    metadata={'type': 'layout_fallback'},
                    hierarchy=[],
                    start_pos=0,
                    end_pos=len(current_chunk)
                ))
                current_chunk = para
                chunk_id += 1
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        
        if current_chunk:
            chunks.append(Chunk(
                content=current_chunk.strip(),
                chunk_id=f"fallback_{chunk_id}",
                metadata={'type': 'layout_fallback'},
                hierarchy=[],
                start_pos=0,
                end_pos=len(current_chunk)
            ))
        
        return chunks
