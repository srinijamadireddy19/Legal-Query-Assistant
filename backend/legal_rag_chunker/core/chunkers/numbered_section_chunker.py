"""
Numbered Section Chunker - Handles documents with numbered sections
Patterns: "1. TITLE", "1.1 Subsection", etc.
"""

import re
from typing import List, Dict, Optional, Tuple
from ..chunker_factory import BaseChunker, Chunk, DetectionResult


class NumberedSectionChunker(BaseChunker):
    """
    Chunks documents with numbered section structure
    Maintains hierarchy: Section -> Subsection -> Content
    """
    
    def _default_config(self) -> Dict:
        return {
            'max_chunk_size': 1500,  # Characters
            'min_chunk_size': 100,
            'overlap': 50,
            'preserve_context': True,  # Include parent section context
            'max_hierarchy_depth': 4,
        }
    
    def chunk(self, text: str, detection_result: DetectionResult) -> List[Chunk]:
        """
        Chunk document by numbered sections and subsections
        
        Algorithm:
        1. Extract section hierarchy
        2. Build section tree
        3. Create chunks respecting hierarchy
        4. Add metadata and breadcrumbs
        """
        text = self._clean_text(text)
        
        # Extract all numbered sections
        sections = self._extract_sections(text)
        
        # Build hierarchy tree
        section_tree = self._build_hierarchy(sections)
        
        # Generate chunks from tree
        chunks = self._generate_chunks_from_tree(section_tree, text)
        
        # Post-process: merge small chunks if needed
        if self.config.get('min_chunk_size'):
            chunks = self._merge_small_chunks(chunks, self.config['min_chunk_size'])
        
        return chunks
    
    def _extract_sections(self, text: str) -> List[Dict]:
        """
        Extract all numbered sections with positions
        
        Returns list of dicts with:
        - number: section number (e.g., "1", "1.1", "1.1.1")
        - title: section title
        - start: character position
        - level: hierarchy level
        """
        sections = []
        
        # Pattern for numbered sections: "1. TITLE" or "1.1 Title"
        pattern = r'^(\d+(?:\.\d+)*)\.\s+([A-Z][^\n]+)'
        
        for match in re.finditer(pattern, text, re.MULTILINE):
            number = match.group(1)
            title = match.group(2).strip()
            start_pos = match.start()
            
            # Calculate hierarchy level from number
            level = number.count('.') + 1
            
            sections.append({
                'number': number,
                'title': title,
                'start': start_pos,
                'level': level,
                'full_header': match.group(0)
            })
        
        return sections
    
    def _build_hierarchy(self, sections: List[Dict]) -> Dict:
        """
        Build a tree structure from flat section list
        
        Returns nested dict representing section hierarchy
        """
        if not sections:
            return {}
        
        root = {'children': [], 'number': '0', 'title': 'ROOT', 'level': 0}
        stack = [root]
        
        for section in sections:
            # Pop stack until we find the parent level
            while stack and stack[-1]['level'] >= section['level']:
                stack.pop()
            
            # Add section as child of current parent
            parent = stack[-1] if stack else root
            section['children'] = []
            parent['children'].append(section)
            
            # Push current section to stack
            stack.append(section)
        
        return root
    
    def _generate_chunks_from_tree(
        self, 
        tree: Dict, 
        full_text: str
    ) -> List[Chunk]:
        """
        Generate chunks by traversing section tree
        Maintains hierarchy breadcrumbs
        """
        chunks = []
        
        def traverse(node: Dict, hierarchy: List[str], parent_content: str = ""):
            """Recursively traverse tree and create chunks"""
            
            # Handle empty tree
            if not node or 'number' not in node:
                return
            
            if node['number'] == '0':  # Root node
                for child in node['children']:
                    traverse(child, hierarchy, parent_content)
                return
            
            # Build current hierarchy path
            current_hierarchy = hierarchy + [f"{node['number']}. {node['title']}"]
            
            # Extract content for this section
            start = node['start']
            
            # Find end position (start of next sibling or end of text)
            end = self._find_section_end(node, full_text)
            
            section_content = full_text[start:end].strip()
            
            # Extract just the content (without subsections)
            own_content = self._extract_own_content(section_content, node.get('children', []))
            
            # Create chunk if there's content
            if own_content and len(own_content) > 50:
                chunk_id = f"section_{node['number'].replace('.', '_')}"
                
                # Optionally include parent context
                if self.config.get('preserve_context') and parent_content:
                    context = f"[Context: {parent_content[:200]}...]\n\n"
                    full_content = context + own_content
                else:
                    full_content = own_content
                
                # Split if too large
                if len(full_content) > self.config['max_chunk_size']:
                    sub_chunks = self._split_large_content(
                        full_content, 
                        chunk_id, 
                        current_hierarchy,
                        start
                    )
                    chunks.extend(sub_chunks)
                else:
                    chunk = Chunk(
                        content=full_content,
                        chunk_id=chunk_id,
                        metadata={
                            'section_number': node['number'],
                            'section_title': node['title'],
                            'level': node['level'],
                            'type': 'numbered_section'
                        },
                        hierarchy=current_hierarchy,
                        start_pos=start,
                        end_pos=end
                    )
                    chunks.append(chunk)
            
            # Recursively process children
            for child in node.get('children', []):
                traverse(child, current_hierarchy, own_content[:500])
        
        traverse(tree, [])
        return chunks
    
    def _find_section_end(self, node: Dict, text: str) -> int:
        """Find where this section ends in the text"""
        # If has children, end is at first child's start
        if node.get('children'):
            return node['children'][0]['start']
        
        # Otherwise, find next section at same or higher level
        current_level = node['level']
        current_pos = node['start']
        
        # Pattern to find next section
        pattern = r'^(\d+(?:\.\d+)*)\.\s+[A-Z]'
        
        for match in re.finditer(pattern, text[current_pos + 10:], re.MULTILINE):
            next_number = match.group(1)
            next_level = next_number.count('.') + 1
            
            if next_level <= current_level:
                return current_pos + 10 + match.start()
        
        return len(text)
    
    def _extract_own_content(self, section_text: str, children: List[Dict]) -> str:
        """Extract content that belongs to this section only (not subsections)"""
        if not children:
            return section_text
        
        # Remove header line
        lines = section_text.split('\n')
        content_lines = lines[1:] if len(lines) > 1 else lines
        
        # Find where first child starts
        first_child = children[0]
        child_marker = f"{first_child['number']}. {first_child['title']}"
        
        own_text = '\n'.join(content_lines)
        if child_marker in own_text:
            own_text = own_text.split(child_marker)[0]
        
        return own_text.strip()
    
    def _split_large_content(
        self, 
        content: str, 
        base_id: str, 
        hierarchy: List[str],
        start_pos: int
    ) -> List[Chunk]:
        """Split large content into smaller chunks with overlap"""
        chunks = []
        max_size = self.config['max_chunk_size']
        overlap = self.config.get('overlap', 50)
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        current_chunk = ""
        chunk_num = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > max_size and current_chunk:
                # Save current chunk
                chunks.append(Chunk(
                    content=current_chunk.strip(),
                    chunk_id=f"{base_id}_part{chunk_num}",
                    metadata={
                        'type': 'numbered_section',
                        'is_partial': True,
                        'part_number': chunk_num
                    },
                    hierarchy=hierarchy,
                    start_pos=start_pos,
                    end_pos=start_pos + len(current_chunk)
                ))
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
                chunk_num += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add final chunk
        if current_chunk:
            chunks.append(Chunk(
                content=current_chunk.strip(),
                chunk_id=f"{base_id}_part{chunk_num}",
                metadata={
                    'type': 'numbered_section',
                    'is_partial': chunk_num > 0,
                    'part_number': chunk_num
                },
                hierarchy=hierarchy,
                start_pos=start_pos,
                end_pos=start_pos + len(content)
            ))
        
        return chunks
