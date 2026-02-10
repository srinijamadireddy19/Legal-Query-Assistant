"""
Statute Chunker - Handles statute and article-based documents
Patterns: "Article 12", "Section 42", "§ 5"
"""

import re
from typing import List, Dict, Optional, Tuple
from ..chunker_factory import BaseChunker, Chunk, DetectionResult


class StatuteChunker(BaseChunker):
    """
    Chunks legal statutes with Article/Section/Clause hierarchy
    Example: Article 12 -> Clause (1) -> Sub-clause (a)
    """
    
    def _default_config(self) -> Dict:
        return {
            'max_chunk_size': 1200,
            'min_chunk_size': 100,
            'overlap': 30,
            'chunk_by_clause': True,  # Chunk at clause level, not just article
            'preserve_article_context': True,
        }
    
    def chunk(self, text: str, detection_result: DetectionResult) -> List[Chunk]:
        """
        Chunk statute document maintaining article/clause hierarchy
        """
        text = self._clean_text(text)
        
        # Extract articles/sections
        articles = self._extract_articles(text)
        
        # For each article, extract clauses
        chunks = []
        for article in articles:
            article_chunks = self._chunk_article(article, text)
            chunks.extend(article_chunks)
        
        # Merge small chunks
        if self.config.get('min_chunk_size'):
            chunks = self._merge_small_chunks(chunks, self.config['min_chunk_size'])
        
        return chunks
    
    def _extract_articles(self, text: str) -> List[Dict]:
        """
        Extract all articles/sections from text
        
        Handles patterns:
        - Article 12
        - Art. 5
        - Section 42
        - § 7
        """
        articles = []
        
        patterns = [
            (r'Article\s+(\d+)', 'Article'),
            (r'Art\.\s+(\d+)', 'Article'),
            (r'Section\s+(\d+)', 'Section'),
            (r'§\s*(\d+)', 'Section'),
        ]
        
        for pattern, article_type in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                number = match.group(1)
                start_pos = match.start()
                
                # Try to extract title (often on same line or next)
                title = self._extract_article_title(text, match.end())
                
                articles.append({
                    'type': article_type,
                    'number': number,
                    'title': title,
                    'start': start_pos,
                    'full_header': match.group(0)
                })
        
        # Sort by position and remove duplicates
        articles.sort(key=lambda x: x['start'])
        articles = self._deduplicate_articles(articles)
        
        return articles
    
    def _extract_article_title(self, text: str, pos: int) -> str:
        """Extract article title following the article number"""
        # Look at next 200 chars
        snippet = text[pos:pos+200]
        
        # Title is typically: "- Title text" or ": Title text" or just "Title text"
        # until next newline
        match = re.search(r'[:\-]?\s*([A-Z][^\n]+)', snippet)
        if match:
            return match.group(1).strip()
        
        return ""
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate article detections (same position)"""
        seen_positions = set()
        unique_articles = []
        
        for article in articles:
            if article['start'] not in seen_positions:
                seen_positions.add(article['start'])
                unique_articles.append(article)
        
        return unique_articles
    
    def _chunk_article(self, article: Dict, full_text: str) -> List[Chunk]:
        """
        Chunk a single article, potentially breaking by clauses
        """
        # Extract article content
        start = article['start']
        end = self._find_article_end(article, full_text)
        article_text = full_text[start:end]
        
        # Build hierarchy breadcrumb
        hierarchy = [f"{article['type']} {article['number']}: {article['title']}"]
        
        # Check if we should chunk by clauses
        if self.config.get('chunk_by_clause'):
            clauses = self._extract_clauses(article_text)
            
            if clauses:
                return self._chunk_by_clauses(
                    article, 
                    article_text, 
                    clauses, 
                    hierarchy,
                    start
                )
        
        # No clauses or disabled - chunk entire article
        return self._chunk_entire_article(article, article_text, hierarchy, start, end)
    
    def _find_article_end(self, article: Dict, text: str) -> int:
        """Find where this article ends (next article starts)"""
        current_pos = article['start']
        
        # Look for next article/section
        patterns = [
            r'Article\s+\d+',
            r'Art\.\s+\d+',
            r'Section\s+\d+',
            r'§\s*\d+',
        ]
        
        next_positions = []
        for pattern in patterns:
            for match in re.finditer(pattern, text[current_pos + 10:], re.IGNORECASE):
                next_positions.append(current_pos + 10 + match.start())
        
        if next_positions:
            return min(next_positions)
        
        return len(text)
    
    def _extract_clauses(self, article_text: str) -> List[Dict]:
        """
        Extract clauses within an article
        
        Patterns:
        - (1), (2), (3)
        - Clause 1, Clause 2
        - Para. 1, Para. 2
        - (a), (b), (c)
        """
        clauses = []
        
        patterns = [
            (r'\((\d+)\)', 'numeric'),      # (1), (2)
            (r'Clause\s+(\d+)', 'clause'),  # Clause 1
            (r'Para\.\s+(\d+)', 'para'),    # Para. 1
            (r'\(([a-z])\)', 'letter'),     # (a), (b)
        ]
        
        for pattern, clause_type in patterns:
            for match in re.finditer(pattern, article_text):
                identifier = match.group(1)
                start_pos = match.start()
                
                clauses.append({
                    'type': clause_type,
                    'identifier': identifier,
                    'start': start_pos,
                    'marker': match.group(0)
                })
        
        # Sort by position
        clauses.sort(key=lambda x: x['start'])
        
        # Filter out nested clauses if we have main clauses
        clauses = self._filter_nested_clauses(clauses)
        
        return clauses
    
    def _filter_nested_clauses(self, clauses: List[Dict]) -> List[Dict]:
        """Keep only main-level clauses (numeric typically)"""
        if not clauses:
            return clauses
        
        # If we have numeric clauses, prefer those
        numeric_clauses = [c for c in clauses if c['type'] == 'numeric']
        if len(numeric_clauses) >= 3:
            return numeric_clauses
        
        # Otherwise keep clause-type
        clause_types = [c for c in clauses if c['type'] in ['clause', 'para']]
        if clause_types:
            return clause_types
        
        return clauses
    
    def _chunk_by_clauses(
        self,
        article: Dict,
        article_text: str,
        clauses: List[Dict],
        hierarchy: List[str],
        article_start: int
    ) -> List[Chunk]:
        """Create chunks for each clause within the article"""
        chunks = []
        
        # Extract preamble (text before first clause)
        preamble = article_text[:clauses[0]['start']].strip() if clauses else ""
        
        for i, clause in enumerate(clauses):
            # Extract clause content
            clause_start = clause['start']
            clause_end = clauses[i+1]['start'] if i+1 < len(clauses) else len(article_text)
            clause_text = article_text[clause_start:clause_end].strip()
            
            # Build full content with optional preamble
            if self.config.get('preserve_article_context') and preamble:
                full_content = f"[Article Context: {preamble[:200]}...]\n\n{clause_text}"
            else:
                full_content = clause_text
            
            # Build hierarchy
            clause_hierarchy = hierarchy + [f"Clause {clause['identifier']}"]
            
            # Create chunk
            chunk_id = f"article_{article['number']}_clause_{clause['identifier']}"
            
            chunk = Chunk(
                content=full_content,
                chunk_id=chunk_id,
                metadata={
                    'article_number': article['number'],
                    'article_title': article['title'],
                    'clause_id': clause['identifier'],
                    'clause_type': clause['type'],
                    'type': 'statute_clause'
                },
                hierarchy=clause_hierarchy,
                start_pos=article_start + clause_start,
                end_pos=article_start + clause_end
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_entire_article(
        self,
        article: Dict,
        article_text: str,
        hierarchy: List[str],
        start: int,
        end: int
    ) -> List[Chunk]:
        """Chunk entire article as one or more chunks if too large"""
        chunks = []
        
        # If small enough, single chunk
        if len(article_text) <= self.config['max_chunk_size']:
            chunk = Chunk(
                content=article_text,
                chunk_id=f"article_{article['number']}",
                metadata={
                    'article_number': article['number'],
                    'article_title': article['title'],
                    'type': 'statute_article'
                },
                hierarchy=hierarchy,
                start_pos=start,
                end_pos=end
            )
            chunks.append(chunk)
        else:
            # Split by paragraphs
            sub_chunks = self._split_large_content(
                article_text,
                f"article_{article['number']}",
                hierarchy,
                start,
                article['number'],
                article['title']
            )
            chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_large_content(
        self,
        content: str,
        base_id: str,
        hierarchy: List[str],
        start_pos: int,
        article_num: str,
        article_title: str
    ) -> List[Chunk]:
        """Split large article content with overlap"""
        chunks = []
        max_size = self.config['max_chunk_size']
        overlap = self.config.get('overlap', 30)
        
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        current_chunk = ""
        chunk_num = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > max_size and current_chunk:
                chunks.append(Chunk(
                    content=current_chunk.strip(),
                    chunk_id=f"{base_id}_part{chunk_num}",
                    metadata={
                        'article_number': article_num,
                        'article_title': article_title,
                        'type': 'statute_article',
                        'is_partial': True,
                        'part_number': chunk_num
                    },
                    hierarchy=hierarchy,
                    start_pos=start_pos,
                    end_pos=start_pos + len(current_chunk)
                ))
                
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
                chunk_num += 1
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        
        if current_chunk:
            chunks.append(Chunk(
                content=current_chunk.strip(),
                chunk_id=f"{base_id}_part{chunk_num}",
                metadata={
                    'article_number': article_num,
                    'article_title': article_title,
                    'type': 'statute_article',
                    'is_partial': chunk_num > 0,
                    'part_number': chunk_num
                },
                hierarchy=hierarchy,
                start_pos=start_pos,
                end_pos=start_pos + len(content)
            ))
        
        return chunks
