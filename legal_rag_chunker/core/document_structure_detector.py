"""
Document Structure Detector - Main orchestrator for hierarchical chunking
Implements a staged detection pipeline for legal documents
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class DocumentType(Enum):
    """Enumeration of detected document types"""
    NUMBERED_SECTIONS = "numbered_sections"
    STATUTE = "statute"
    POLICY_OR_TERMS = "policy_or_terms"
    LAYOUT_BASED = "layout_based"
    UNSTRUCTURED = "unstructured"


@dataclass
class DetectionResult:
    """Result from document structure detection"""
    doc_type: DocumentType
    confidence: float
    metadata: Dict
    structure_hints: Dict


class DocumentStructureDetector:
    """
    Main orchestrator that runs through detection stages sequentially
    Each stage can veto and pass control to the next stage
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize detector with optional configuration
        
        Args:
            config: Optional configuration dictionary for thresholds and patterns
        """
        self.config = config or self._default_config()
        self.detection_stages = [
            self._detect_numbered_sections,
            self._detect_statute_pattern,
            self._detect_list_pattern,
            self._detect_layout_based,
            self._fallback_unstructured
        ]
    
    def _default_config(self) -> Dict:
        """Default configuration for detection thresholds"""
        return {
            'numbered_section_threshold': 3,  # Min occurrences to confirm pattern
            'statute_threshold': 2,
            'list_pattern_threshold': 5,
            'confidence_thresholds': {
                'high': 0.8,
                'medium': 0.5,
                'low': 0.3
            }
        }
    
    def detect(self, text: str, layout_info: Optional[Dict] = None) -> DetectionResult:
        """
        Main detection method - runs through all stages sequentially
        
        Args:
            text: Raw document text
            layout_info: Optional layout information from PDF parser
            
        Returns:
            DetectionResult with type, confidence, and metadata
        """
        for stage in self.detection_stages:
            result = stage(text, layout_info)
            if result is not None:
                return result
        
        # Should never reach here due to fallback, but just in case
        return DetectionResult(
            doc_type=DocumentType.UNSTRUCTURED,
            confidence=1.0,
            metadata={},
            structure_hints={}
        )
    
    def _detect_numbered_sections(
        self, 
        text: str, 
        layout_info: Optional[Dict]
    ) -> Optional[DetectionResult]:
        """
        STAGE 1: Detect numbered section patterns
        Patterns: "1. TITLE", "2. TITLE", "1.1 Subsection"
        """
        # Pattern variations for numbered sections
        patterns = [
            r'\n\d+\.\s+[A-Z][A-Z\s]{2,}',  # "1. SECTION TITLE"
            r'\n\d+\.\d+\s+[A-Z]',           # "1.1 Subsection"
            r'^\d+\.\s+[A-Z][A-Z\s]{2,}',    # At start of line
        ]
        
        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.MULTILINE))
        
        threshold = self.config['numbered_section_threshold']
        
        if len(matches) >= threshold:
            # Analyze numbering structure
            numbers = [int(re.search(r'\d+', m.group()).group()) for m in matches]
            is_sequential = self._check_sequential(numbers)
            
            # Detect subsection depth
            subsection_pattern = r'\d+\.\d+\.?\d*'
            subsections = re.findall(subsection_pattern, text)
            max_depth = max([s.count('.') for s in subsections]) if subsections else 1
            
            confidence = min(0.9, 0.6 + (len(matches) * 0.05))
            
            return DetectionResult(
                doc_type=DocumentType.NUMBERED_SECTIONS,
                confidence=confidence,
                metadata={
                    'pattern_count': len(matches),
                    'is_sequential': is_sequential,
                    'max_depth': max_depth
                },
                structure_hints={
                    'primary_delimiter': 'numbered_section',
                    'secondary_delimiter': 'subsection',
                    'hierarchy_depth': max_depth
                }
            )
        
        return None
    
    def _detect_statute_pattern(
        self, 
        text: str, 
        layout_info: Optional[Dict]
    ) -> Optional[DetectionResult]:
        """
        STAGE 2: Detect statute/article patterns
        Patterns: "Article 12", "Art. 5", "Section 42"
        """
        patterns = [
            (r'Article\s+\d+', 'article'),
            (r'Art\.\s+\d+', 'article_abbrev'),
            (r'Section\s+\d+', 'section'),
            (r'§\s*\d+', 'section_symbol'),
        ]
        
        all_matches = []
        pattern_types = []
        
        for pattern, ptype in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            all_matches.extend(matches)
            if matches:
                pattern_types.append(ptype)
        
        threshold = self.config['statute_threshold']
        
        if len(all_matches) >= threshold:
            # Detect clause patterns under articles
            clause_patterns = r'Clause\s+\d+|Para\.\s+\d+|\(\d+\)'
            clauses = re.findall(clause_patterns, text, re.IGNORECASE)
            
            confidence = min(0.85, 0.5 + (len(all_matches) * 0.1))
            
            return DetectionResult(
                doc_type=DocumentType.STATUTE,
                confidence=confidence,
                metadata={
                    'pattern_count': len(all_matches),
                    'pattern_types': pattern_types,
                    'clause_count': len(clauses)
                },
                structure_hints={
                    'primary_delimiter': 'article',
                    'secondary_delimiter': 'clause',
                    'hierarchy_depth': 2
                }
            )
        
        return None
    
    def _detect_list_pattern(
        self, 
        text: str, 
        layout_info: Optional[Dict]
    ) -> Optional[DetectionResult]:
        """
        STAGE 3: Detect heavy list/enumeration usage
        Patterns: "(a)", "(b)", bullet points, dashes
        """
        patterns = [
            (r'\([a-z]\)', 'letter_paren'),
            (r'\([ivx]+\)', 'roman_paren'),
            (r'\n\s*[•●○]\s', 'bullet'),
            (r'\n\s*-\s+[A-Z]', 'dash'),
        ]
        
        all_matches = []
        pattern_types = {}
        
        for pattern, ptype in patterns:
            matches = list(re.finditer(pattern, text))
            all_matches.extend(matches)
            pattern_types[ptype] = len(matches)
        
        threshold = self.config['list_pattern_threshold']
        
        if len(all_matches) >= threshold:
            # Calculate list density
            lines = text.split('\n')
            list_density = len(all_matches) / max(len(lines), 1)
            
            # Dominant pattern type
            dominant_type = max(pattern_types.items(), key=lambda x: x[1])[0] if pattern_types else None
            
            confidence = min(0.75, 0.4 + (list_density * 10))
            
            return DetectionResult(
                doc_type=DocumentType.POLICY_OR_TERMS,
                confidence=confidence,
                metadata={
                    'pattern_count': len(all_matches),
                    'pattern_types': pattern_types,
                    'list_density': list_density,
                    'dominant_type': dominant_type
                },
                structure_hints={
                    'primary_delimiter': 'paragraph',
                    'secondary_delimiter': 'list_item',
                    'hierarchy_depth': 2
                }
            )
        
        return None
    
    def _detect_layout_based(
        self, 
        text: str, 
        layout_info: Optional[Dict]
    ) -> Optional[DetectionResult]:
        """
        STAGE 4: Use layout information if available
        Looks for headings based on font size, bold, centering
        """
        if not layout_info:
            return None
        
        # Extract potential headings from layout info
        headings = layout_info.get('headings', [])
        font_changes = layout_info.get('font_changes', [])
        
        if len(headings) >= 3:  # At least 3 headings detected
            # Analyze heading hierarchy
            heading_levels = self._analyze_heading_levels(headings)
            
            confidence = min(0.8, 0.5 + (len(headings) * 0.05))
            
            return DetectionResult(
                doc_type=DocumentType.LAYOUT_BASED,
                confidence=confidence,
                metadata={
                    'heading_count': len(headings),
                    'heading_levels': heading_levels,
                    'font_changes': len(font_changes)
                },
                structure_hints={
                    'primary_delimiter': 'heading',
                    'secondary_delimiter': 'paragraph',
                    'hierarchy_depth': len(heading_levels)
                }
            )
        
        return None
    
    def _fallback_unstructured(
        self, 
        text: str, 
        layout_info: Optional[Dict]
    ) -> DetectionResult:
        """
        STAGE 5: Fallback for unstructured documents
        Always returns a result (last resort)
        """
        # Analyze basic structure even for unstructured docs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        avg_para_length = sum(len(p) for p in paragraphs) / max(len(paragraphs), 1)
        
        return DetectionResult(
            doc_type=DocumentType.UNSTRUCTURED,
            confidence=1.0,  # We're certain it's unstructured
            metadata={
                'paragraph_count': len(paragraphs),
                'avg_paragraph_length': avg_para_length
            },
            structure_hints={
                'primary_delimiter': 'paragraph',
                'secondary_delimiter': None,
                'hierarchy_depth': 1,
                'fallback': True
            }
        )
    
    # Helper methods
    
    def _check_sequential(self, numbers: List[int]) -> bool:
        """Check if numbers are roughly sequential"""
        if len(numbers) < 2:
            return True
        
        sorted_nums = sorted(set(numbers))
        gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
        avg_gap = sum(gaps) / len(gaps)
        
        return avg_gap <= 2  # Allow some missing numbers
    
    def _analyze_heading_levels(self, headings: List[Dict]) -> Dict[int, int]:
        """Analyze heading hierarchy from layout info"""
        levels = {}
        for heading in headings:
            level = heading.get('level', 1)
            levels[level] = levels.get(level, 0) + 1
        return levels
