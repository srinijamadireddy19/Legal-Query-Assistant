"""
LLM Pipeline
────────────
Main pipeline for generating responses with RAG.
Handles conversation state, context retrieval, and response generation.

Usage:
    from llm_pipeline import LLMPipeline, LLMConfig
    from embedding_pipeline import EmbeddingPipeline
    from vector_storage import VectorStoragePipeline
    
    # Initialize
    llm_config = LLMConfig(api_key="your-groq-key")
    llm = LLMPipeline(llm_config, embedder, storage)
    
    # Ask questions
    response = llm.query(
        query="What are the payment terms?",
        collection_name="legal_docs",
    )
    
    print(response.answer)
"""

import logging
import time
from typing import List, Optional, Any

from .core.models import (
    LLMConfig,
    LLMResponse,
    QueryContext,
    QueryType,
    ContextQuality,
    ConversationMessage,
    RetrievedContext,
)
from .core.classifier import QueryClassifier
from .prompts.templates import PromptSelector
from .providers.groq_provider import GroqProvider

log = logging.getLogger(__name__)


class LLMPipeline:
    """
    Complete LLM pipeline with RAG.
    Manages conversation, retrieves context, generates responses.
    """
    
    def __init__(
        self,
        config: LLMConfig,
        embedder: Any,  # EmbeddingPipeline instance
        storage: Any,   # VectorStoragePipeline instance
    ):
        """
        Initialize LLM pipeline.
        
        Args:
            config: LLMConfig with model settings
            embedder: EmbeddingPipeline for query embedding
            storage: VectorStoragePipeline for context retrieval
        """
        self.config = config
        self.embedder = embedder
        self.storage = storage
        
        # Initialize LLM provider
        if config.provider == "groq":
            self.provider = GroqProvider(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        # Conversation state
        self.history: List[ConversationMessage] = []
        
        # Classifier
        self.classifier = QueryClassifier()
        
        log.info(f"LLM Pipeline initialized ({config.provider}: {config.model_name})")
    
    # ══════════════════════════════════════════════════════════════════════
    # Main Query Method
    # ══════════════════════════════════════════════════════════════════════
    
    def query(
        self,
        query: str,
        collection_name: str,
        k: int = None,
        filter_by: str = None,
    ) -> LLMResponse:
        """
        Process a user query and generate response.
        
        Args:
            query: User's question
            collection_name: Collection to search
            k: Number of contexts to retrieve (default: from config)
            filter_by: Optional Typesense filter
            
        Returns:
            LLMResponse with answer and metadata
        """
        start_time = time.time()
        
        # 1. Classify query type
        query_type = self.classifier.classify(query, self.history)
        log.info(f"Query type: {query_type.value}")
        
        # 2. Retrieve context from vector DB
        contexts = self._retrieve_context(
            query=query,
            collection_name=collection_name,
            k=k or self.config.max_context_length,
            filter_by=filter_by,
        )
        
        # 3. Build query context
        query_ctx = QueryContext(
            query=query,
            query_type=query_type,
            contexts=contexts,
            history=self.history.copy(),
            collection_name=collection_name,
            num_contexts_requested=k or self.config.max_context_length,
        )
        
        log.info(
            f"Retrieved {len(contexts)} contexts, "
            f"quality: {query_ctx.context_quality.value}, "
            f"best score: {query_ctx.best_score:.3f}"
        )
        
        # 4. Generate response
        answer, metadata = self._generate_response(query_ctx)
        
        # 5. Build LLM response
        response = LLMResponse(
            answer=answer,
            query=query,
            contexts_used=contexts,
            context_quality=query_ctx.context_quality,
            citations=query_ctx.get_citations(),
            model_name=self.config.model_name,
            query_type=query_type,
            prompt_tokens=metadata.get("prompt_tokens", 0),
            completion_tokens=metadata.get("completion_tokens", 0),
            total_tokens=metadata.get("total_tokens", 0),
            response_time_sec=time.time() - start_time,
        )
        
        # 6. Update conversation history
        self._update_history(query, answer)
        
        return response
    
    # ══════════════════════════════════════════════════════════════════════
    # Context Retrieval
    # ══════════════════════════════════════════════════════════════════════
    
    def _retrieve_context(
        self,
        query: str,
        collection_name: str,
        k: int,
        filter_by: str = None,
    ) -> List[RetrievedContext]:
        """Retrieve relevant contexts from vector DB."""
        try:
            # Embed query
            query_embedding = self.embedder.embedder.embed_single(query)
            
            if not query_embedding:
                log.error("Failed to embed query")
                return []
            
            # Search vector DB
            results = self.storage.search(
                collection_name=collection_name,
                query_embedding=query_embedding,
                k=k,
                filter_by=filter_by,
            )
            
            # Filter by threshold
            contexts = []
            for r in results:
                if r['score'] >= self.config.context_score_threshold:
                    ctx = RetrievedContext(
                        text=r['text'],
                        score=r['score'],
                        source_file=r.get('source_file', ''),
                        hierarchy=r.get('hierarchy', []),
                        chunk_id=r['id'],
                        page_numbers=r.get('page_numbers', []),
                        metadata=r.get('metadata', {}),
                    )
                    contexts.append(ctx)
            
            return contexts
            
        except Exception as e:
            log.error(f"Context retrieval failed: {e}")
            return []
    
    # ══════════════════════════════════════════════════════════════════════
    # Response Generation
    # ══════════════════════════════════════════════════════════════════════
    
    def _generate_response(self, query_ctx: QueryContext) -> tuple[str, dict]:
        """
        Generate response using LLM.
        
        Returns:
            (answer_text, metadata_dict)
        """
        # Select appropriate prompt
        user_prompt = PromptSelector.select_prompt(
            query_ctx, 
            concise=self.config.be_concise
        )
        
        system_prompt = PromptSelector.get_system_prompt(
            concise=self.config.be_concise
        )
        
        # Generate with LLM
        try:
            result = self.provider.chat(
                user_message=user_prompt,
                system_prompt=system_prompt,
                history=[],  # History is already in the prompt
            )
            
            return result["content"], result
            
        except Exception as e:
            log.error(f"LLM generation failed: {e}")
            
            # Fallback response
            if query_ctx.context_quality == ContextQuality.NO_CONTEXT:
                fallback = (
                    "I apologize, but I couldn't find relevant information in the "
                    "available documents to answer your question. Please try:\n"
                    "- Rephrasing your question\n"
                    "- Being more specific about what you're looking for\n"
                    "- Checking if the relevant documents have been indexed"
                )
            else:
                fallback = (
                    "I apologize, but I encountered an error generating a response. "
                    "Please try asking your question again."
                )
            
            return fallback, {"error": str(e)}
    
    # ══════════════════════════════════════════════════════════════════════
    # Conversation Management
    # ══════════════════════════════════════════════════════════════════════
    
    def _update_history(self, user_query: str, assistant_answer: str):
        """Update conversation history."""
        self.history.append(
            ConversationMessage(role="user", content=user_query)
        )
        self.history.append(
            ConversationMessage(role="assistant", content=assistant_answer)
        )
        
        # Trim history if too long
        max_messages = self.config.max_history_messages
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
        log.info("Conversation history cleared")
    
    def get_history(self) -> List[ConversationMessage]:
        """Get conversation history."""
        return self.history.copy()
    
    # ══════════════════════════════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════════════════════════════
    
    def __repr__(self) -> str:
        return (
            f"LLMPipeline("
            f"provider={self.config.provider}, "
            f"model={self.config.model_name}, "
            f"history_len={len(self.history)})"
        )


# ══════════════════════════════════════════════════════════════════════════
# Conversation Helper
# ══════════════════════════════════════════════════════════════════════════

class Conversation:
    """
    Helper class for managing multi-turn conversations.
    Wraps LLMPipeline with convenience methods.
    """
    
    def __init__(
        self,
        llm_pipeline: LLMPipeline,
        collection_name: str,
    ):
        """
        Initialize conversation.
        
        Args:
            llm_pipeline: LLMPipeline instance
            collection_name: Default collection to search
        """
        self.llm = llm_pipeline
        self.collection_name = collection_name
    
    def ask(self, question: str, **kwargs) -> str:
        """
        Ask a question and get the answer.
        
        Args:
            question: User question
            **kwargs: Additional args for llm.query()
            
        Returns:
            Answer text
        """
        response = self.llm.query(
            query=question,
            collection_name=self.collection_name,
            **kwargs
        )
        return response.answer
    
    def ask_with_citations(self, question: str, **kwargs) -> str:
        """Ask and get answer with citations appended."""
        response = self.llm.query(
            query=question,
            collection_name=self.collection_name,
            **kwargs
        )
        return response.format_with_citations()
    
    def reset(self):
        """Start new conversation."""
        self.llm.clear_history()