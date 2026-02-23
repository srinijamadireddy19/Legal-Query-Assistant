"""
Core data models for LLM pipeline.
Handles queries, contexts, responses, and conversation state.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class QueryType(Enum):
    """Type of user query."""
    NEW = "new"                    # First question in conversation
    FOLLOWUP = "followup"          # Follow-up question
    CLARIFICATION = "clarification"  # Asking to clarify previous answer
    RELATED = "related"            # Related but new topic


class ContextQuality(Enum):
    """Quality of retrieved context."""
    HIGH = "high"           # Score > 0.7
    MEDIUM = "medium"       # Score 0.5-0.7
    LOW = "low"            # Score 0.3-0.5
    NO_CONTEXT = "none"    # Score < 0.3 or no results


@dataclass
class RetrievedContext:
    """
    Context retrieved from vector store.
    Includes text, score, and metadata.
    """
    text: str
    score: float
    source_file: str
    hierarchy: List[str]
    chunk_id: str
    page_numbers: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def quality(self) -> ContextQuality:
        """Determine quality based on similarity score."""
        if self.score >= 0.7:
            return ContextQuality.HIGH
        elif self.score >= 0.5:
            return ContextQuality.MEDIUM
        elif self.score >= 0.3:
            return ContextQuality.LOW
        else:
            return ContextQuality.NO_CONTEXT
    
    def format_citation(self) -> str:
        """Format as citation string."""
        hier = " > ".join(self.hierarchy) if self.hierarchy else "Document"
        pages = f", p. {','.join(map(str, self.page_numbers))}" if self.page_numbers else ""
        return f"{self.source_file} ({hier}{pages})"


@dataclass
class ConversationMessage:
    """Single message in conversation history."""
    role: str          # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to LLM API format."""
        return {"role": self.role, "content": self.content}


@dataclass
class QueryContext:
    """
    Complete context for answering a query.
    Includes user question, retrieved contexts, and conversation history.
    """
    query: str
    query_type: QueryType
    
    # Retrieved contexts (from vector DB)
    contexts: List[RetrievedContext] = field(default_factory=list)
    
    # Conversation history
    history: List[ConversationMessage] = field(default_factory=list)
    
    # Metadata
    collection_name: str = ""
    num_contexts_requested: int = 5
    
    @property
    def best_score(self) -> float:
        """Get highest similarity score from contexts."""
        return max((c.score for c in self.contexts), default=0.0)
    
    @property
    def context_quality(self) -> ContextQuality:
        """Determine overall context quality."""
        if not self.contexts:
            return ContextQuality.NO_CONTEXT
        
        score = self.best_score
        if score >= 0.7:
            return ContextQuality.HIGH
        elif score >= 0.5:
            return ContextQuality.MEDIUM
        elif score >= 0.3:
            return ContextQuality.LOW
        else:
            return ContextQuality.NO_CONTEXT
    
    @property
    def has_history(self) -> bool:
        """Check if conversation has history."""
        return len(self.history) > 0
    
    def get_context_text(self, max_contexts: int = 5) -> str:
        """
        Format contexts as text for prompt.
        Returns top N contexts concatenated.
        """
        if not self.contexts:
            return ""
        
        formatted = []
        for i, ctx in enumerate(self.contexts[:max_contexts], 1):
            hier = " > ".join(ctx.hierarchy) if ctx.hierarchy else "Document"
            formatted.append(
                f"[Context {i}] (Score: {ctx.score:.2f})\n"
                f"Source: {ctx.source_file}\n"
                f"Section: {hier}\n"
                f"Content: {ctx.text}\n"
            )
        
        return "\n".join(formatted)
    
    def get_citations(self, max_citations: int = 3) -> List[str]:
        """Get formatted citations for response."""
        return [
            ctx.format_citation() 
            for ctx in self.contexts[:max_citations]
            if ctx.score >= 0.5  # Only cite high-quality sources
        ]


@dataclass
class LLMResponse:
    """
    Response from LLM with metadata.
    """
    answer: str
    query: str
    
    # Context used
    contexts_used: List[RetrievedContext] = field(default_factory=list)
    context_quality: ContextQuality = ContextQuality.NO_CONTEXT
    
    # Citations
    citations: List[str] = field(default_factory=list)
    
    # Metadata
    model_name: str = ""
    query_type: QueryType = QueryType.NEW
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    response_time_sec: float = 0.0
    
    # Warnings/info
    fallback_used: bool = False
    warnings: List[str] = field(default_factory=list)
    
    def format_with_citations(self) -> str:
        """Format answer with citations appended."""
        response = self.answer
        
        if self.citations:
            response += "\n\n**Sources:**\n"
            for i, citation in enumerate(self.citations, 1):
                response += f"{i}. {citation}\n"
        
        return response
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "answer": self.answer,
            "query": self.query,
            "context_quality": self.context_quality.value,
            "citations": self.citations,
            "model_name": self.model_name,
            "query_type": self.query_type.value,
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "total": self.total_tokens,
            },
            "response_time_sec": self.response_time_sec,
            "fallback_used": self.fallback_used,
            "warnings": self.warnings,
        }


@dataclass
class LLMConfig:
    """
    Configuration for LLM providers.
    """
    # Provider
    provider: str = "groq"  # "groq", "openai", "anthropic"
    api_key: str = ""
    
    # Model selection
    model_name: str = "llama-3.1-70b-versatile"
    
    # Generation parameters
    temperature: float = 0.1        # Low for factual responses
    max_tokens: int = 1024
    top_p: float = 1.0
    
    # Context settings
    max_context_length: int = 5     # Number of chunks to include
    context_score_threshold: float = 0.3  # Minimum similarity score
    
    # Prompt settings
    use_citations: bool = True
    be_concise: bool = False        # Concise vs detailed responses
    
    # Conversation
    max_history_messages: int = 10  # Keep last N messages
    
    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []
        
        if not self.api_key:
            errors.append("api_key is required")
        
        if self.temperature < 0 or self.temperature > 2:
            errors.append(f"Invalid temperature: {self.temperature}")
        
        if self.max_tokens <= 0:
            errors.append(f"Invalid max_tokens: {self.max_tokens}")
        
        if self.context_score_threshold < 0 or self.context_score_threshold > 1:
            errors.append(f"Invalid context_score_threshold: {self.context_score_threshold}")
        
        return errors