"""
Policy Chunker - Handles policy/terms documents with heavy list usage
Patterns: (a), (b), bullets, enumerated lists
"""

import re
from typing import List, Dict, Optional
from ..chunker_factory import BaseChunker, Chunk, DetectionResult


class PolicyChunker(BaseChunker):
    """
    Chunks policy documents with paragraph + list item hierarchy
    Handles: (a), (b), bullets, roman numerals, etc.
    """
    
    def _default_config(self) -> Dict:
        return {
            'max_chunk_size': 1000,
            'min_chunk_size': 100,
            'keep_list_context': True,  # Include parent paragraph with list
            'group_related_items': True,  # Group related list items
        }
    
    def chunk(self, text: str, detection_result: DetectionResult) -> List[Chunk]:
        """
        Chunk policy document maintaining paragraph-list hierarchy
        """
        text = self._clean_text(text)
        
        # Split into logical blocks (paragraphs with their lists)
        blocks = self._extract_blocks(text)
        
        # Create chunks from blocks
        chunks = []
        for i, block in enumerate(blocks):
            block_chunks = self._chunk_block(block, i)
            chunks.extend(block_chunks)
        
        # Post-process
        if self.config.get('min_chunk_size'):
            chunks = self._merge_small_chunks(chunks, self.config['min_chunk_size'])
        
        return chunks
    
    def _extract_blocks(self, text: str) -> List[Dict]:
        """
        Extract logical blocks: paragraphs with their associated lists
        
        A block consists of:
        - Optional heading
        - Paragraph text
        - Associated list items
        """
        blocks = []
        lines = text.split('\n')
        
        current_block = {
            'heading': None,
            'paragraph': [],
            'list_items': [],
            'start_line': 0
        }
        
        in_list = False
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            # Check if it's a heading (all caps, short, no ending punctuation)
            if self._is_heading(line_stripped):
                # Save previous block if exists
                if current_block['paragraph'] or current_block['list_items']:
                    blocks.append(current_block)
                
                # Start new block
                current_block = {
                    'heading': line_stripped,
                    'paragraph': [],
                    'list_items': [],
                    'start_line': i
                }
                in_list = False
                continue
            
            # Check if it's a list item
            list_marker = self._get_list_marker(line_stripped)
            
            if list_marker:
                in_list = True
                current_block['list_items'].append({
                    'marker': list_marker,
                    'content': self._remove_list_marker(line_stripped, list_marker),
                    'line': i
                })
            else:
                if in_list:
                    # Continuation of previous list item
                    if current_block['list_items']:
                        current_block['list_items'][-1]['content'] += ' ' + line_stripped
                else:
                    # Regular paragraph text
                    current_block['paragraph'].append(line_stripped)
        
        # Add final block
        if current_block['paragraph'] or current_block['list_items']:
            blocks.append(current_block)
        
        return blocks
    
    def _is_heading(self, line: str) -> bool:
        """Detect if line is a heading"""
        # Heuristics for heading detection
        if len(line) > 100:  # Too long for heading
            return False
        
        # Check if mostly uppercase
        if line.isupper() or (sum(c.isupper() for c in line) / max(len(line), 1) > 0.7):
            return True
        
        # Check for numbered headings like "1. TITLE" or "1.1 Title"
        if re.match(r'^\d+\.?\d*\s+[A-Z]', line):
            return True
        
        return False
    
    def _get_list_marker(self, line: str) -> Optional[str]:
        """
        Detect list marker and return it
        
        Patterns:
        - (a), (b), (c)
        - (i), (ii), (iii)
        - (1), (2), (3)
        - •, ●, ○
        - -, *
        """
        patterns = [
            r'^\([a-z]\)\s',       # (a)
            r'^\([ivx]+\)\s',      # (i), (ii)
            r'^\(\d+\)\s',         # (1), (2)
            r'^[•●○]\s',           # Bullets
            r'^[-*]\s+',           # Dash/asterisk
            r'^\d+\.\s',           # 1., 2.
            r'^[a-z]\.\s',         # a., b.
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _remove_list_marker(self, line: str, marker: str) -> str:
        """Remove the list marker from line"""
        return line[len(marker):].strip()
    
    def _chunk_block(self, block: Dict, block_index: int) -> List[Chunk]:
        """
        Create chunks from a single block
        
        Strategy:
        - If small, keep as single chunk
        - If large, split intelligently by list groups
        """
        chunks = []
        
        # Reconstruct block text
        heading = block.get('heading', '')
        paragraph_text = ' '.join(block['paragraph'])
        
        # Build hierarchy
        hierarchy = []
        if heading:
            hierarchy.append(heading)
        
        # Check if we should keep as single chunk
        list_text = self._format_list_items(block['list_items'])
        full_text = self._combine_paragraph_and_list(paragraph_text, list_text, heading)
        
        if len(full_text) <= self.config['max_chunk_size']:
            # Single chunk
            chunk = Chunk(
                content=full_text,
                chunk_id=f"block_{block_index}",
                metadata={
                    'block_index': block_index,
                    'has_heading': bool(heading),
                    'list_item_count': len(block['list_items']),
                    'type': 'policy_block'
                },
                hierarchy=hierarchy,
                start_pos=block['start_line'],
                end_pos=block['start_line'] + len(full_text)
            )
            chunks.append(chunk)
        else:
            # Need to split
            if self.config.get('group_related_items') and block['list_items']:
                # Split by grouping related list items
                chunks.extend(self._chunk_with_grouped_lists(
                    block, 
                    paragraph_text, 
                    hierarchy, 
                    block_index
                ))
            else:
                # Simple split
                chunks.extend(self._simple_split_block(
                    full_text, 
                    hierarchy, 
                    block_index
                ))
        
        return chunks
    
    def _format_list_items(self, list_items: List[Dict]) -> str:
        """Format list items back to text"""
        if not list_items:
            return ""
        
        formatted = []
        for item in list_items:
            formatted.append(f"{item['marker']}{item['content']}")
        
        return '\n'.join(formatted)
    
    def _combine_paragraph_and_list(
        self, 
        paragraph: str, 
        list_text: str,
        heading: str = ""
    ) -> str:
        """Combine paragraph and list into coherent text"""
        parts = []
        
        if heading:
            parts.append(f"{heading}\n")
        
        if paragraph:
            parts.append(paragraph)
        
        if list_text:
            if paragraph:
                parts.append('\n\n')
            parts.append(list_text)
        
        return ''.join(parts)
    
    def _chunk_with_grouped_lists(
        self,
        block: Dict,
        paragraph_text: str,
        hierarchy: List[str],
        block_index: int
    ) -> List[Chunk]:
        """
        Split block by grouping related list items
        Tries to keep semantically related items together
        """
        chunks = []
        list_items = block['list_items']
        heading = block.get('heading', '')
        
        # Group items by semantic breaks (detect topic changes)
        item_groups = self._group_list_items(list_items)
        
        for group_idx, group in enumerate(item_groups):
            # Include paragraph context with first group
            if group_idx == 0 and self.config.get('keep_list_context'):
                context = f"{paragraph_text}\n\n" if paragraph_text else ""
            else:
                context = ""
            
            group_text = self._format_list_items(group)
            full_content = context + group_text
            
            if heading:
                full_content = f"{heading}\n\n{full_content}"
            
            chunk = Chunk(
                content=full_content,
                chunk_id=f"block_{block_index}_group_{group_idx}",
                metadata={
                    'block_index': block_index,
                    'group_index': group_idx,
                    'list_item_count': len(group),
                    'type': 'policy_list_group'
                },
                hierarchy=hierarchy + [f"Items {group[0]['marker']}-{group[-1]['marker']}"],
                start_pos=group[0]['line'],
                end_pos=group[-1]['line']
            )
            chunks.append(chunk)
        
        return chunks
    
    def _group_list_items(self, items: List[Dict]) -> List[List[Dict]]:
        """
        Group list items by semantic similarity
        Simple heuristic: break on marker type change or size limit
        """
        if not items:
            return []
        
        groups = []
        current_group = [items[0]]
        current_size = len(items[0]['content'])
        prev_marker_type = self._get_marker_type(items[0]['marker'])
        
        for item in items[1:]:
            marker_type = self._get_marker_type(item['marker'])
            item_size = len(item['content'])
            
            # Break group if:
            # 1. Marker type changes
            # 2. Current group would be too large
            if (marker_type != prev_marker_type or 
                current_size + item_size > self.config['max_chunk_size'] * 0.8):
                
                groups.append(current_group)
                current_group = [item]
                current_size = item_size
            else:
                current_group.append(item)
                current_size += item_size
            
            prev_marker_type = marker_type
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _get_marker_type(self, marker: str) -> str:
        """Classify marker type"""
        if re.match(r'\([a-z]\)', marker):
            return 'letter'
        elif re.match(r'\([ivx]+\)', marker):
            return 'roman'
        elif re.match(r'\(\d+\)', marker):
            return 'number'
        elif re.match(r'[•●○]', marker):
            return 'bullet'
        else:
            return 'other'
    
    def _simple_split_block(
        self,
        text: str,
        hierarchy: List[str],
        block_index: int
    ) -> List[Chunk]:
        """Simple split when grouping isn't applicable"""
        chunks = []
        max_size = self.config['max_chunk_size']
        
        # Split by sentences or paragraphs
        parts = text.split('\n\n')
        
        current_chunk = ""
        chunk_num = 0
        
        for part in parts:
            if len(current_chunk) + len(part) > max_size and current_chunk:
                chunks.append(Chunk(
                    content=current_chunk.strip(),
                    chunk_id=f"block_{block_index}_part_{chunk_num}",
                    metadata={
                        'block_index': block_index,
                        'is_partial': True,
                        'type': 'policy_block'
                    },
                    hierarchy=hierarchy,
                    start_pos=0,
                    end_pos=len(current_chunk)
                ))
                current_chunk = part
                chunk_num += 1
            else:
                current_chunk = current_chunk + "\n\n" + part if current_chunk else part
        
        if current_chunk:
            chunks.append(Chunk(
                content=current_chunk.strip(),
                chunk_id=f"block_{block_index}_part_{chunk_num}",
                metadata={
                    'block_index': block_index,
                    'is_partial': chunk_num > 0,
                    'type': 'policy_block'
                },
                hierarchy=hierarchy,
                start_pos=0,
                end_pos=len(current_chunk)
            ))
        
        return chunks
