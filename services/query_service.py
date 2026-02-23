"""
Query Service
─────────────
Handles RAG queries with conversation management.
"""

import uuid
import logging
from typing import Dict, Any

from config.settings import PipelineManager
from llm_pipeline.pipeline import LLMPipeline

log = logging.getLogger(__name__)


class QueryService:
    """
    Service for RAG queries.
    Manages conversations and delegates to LLM pipeline.
    """
    
    def __init__(self, pipeline_manager: PipelineManager):
        self.pm = pipeline_manager
    
    async def query(
        self,
        query: str,
        collection_name: str,
        conversation_id: str = None,
        k: int = 5,
        filter_by: str = None,
    ) -> Dict[str, Any]:
        """
        Process a user query through the RAG system.
        
        Args:
            query: User's question
            collection_name: Collection to search
            conversation_id: Optional conversation ID for multi-turn
            k: Number of contexts to retrieve
            filter_by: Optional Typesense filter
            
        Returns:
            Query result dict
        """
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Get or create conversation
        conversation = self.pm.get_conversation(conversation_id)
        
        # Get or create LLM instance for this conversation
        if conversation["llm_instance"] is None:
            # Create new LLM instance with isolated history
            from llm_pipeline import LLMConfig
            
            llm_config = LLMConfig(
                provider="groq",
                api_key=self.pm.llm.config.api_key,
                model_name=self.pm.llm.config.model_name,
                temperature=self.pm.llm.config.temperature,
                max_tokens=self.pm.llm.config.max_tokens,
                context_score_threshold=self.pm.llm.config.context_score_threshold,
                max_context_length=self.pm.llm.config.max_context_length,
            )
            
            llm_instance = LLMPipeline(
                llm_config,
                self.pm.embedder,
                self.pm.storage,
            )
            
            # Restore history if exists
            llm_instance.history = conversation["history"].copy()
            
            conversation["llm_instance"] = llm_instance
        else:
            llm_instance = conversation["llm_instance"]
        
        log.info(
            f"Query: '{query}' (conversation: {conversation_id}, "
            f"collection: {collection_name})"
        )
        
        try:
            # Execute query
            response = llm_instance.query(
                query=query,
                collection_name=collection_name,
                k=k,
                filter_by=filter_by,
            )
            
            # Update conversation history
            conversation["history"] = llm_instance.history.copy()
            
            log.info(
                f"Response: {len(response.answer)} chars, "
                f"type: {response.query_type.value}, "
                f"quality: {response.context_quality.value}, "
                f"contexts: {len(response.contexts_used)}"
            )
            
            # Format response
            return {
                "answer": response.answer,
                "query": query,
                "conversation_id": conversation_id,
                "query_type": response.query_type.value,
                "context_quality": response.context_quality.value,
                "citations": response.citations,
                "contexts_used": len(response.contexts_used),
                "best_score": max(
                    (ctx.score for ctx in response.contexts_used),
                    default=0.0
                ),
                "response_time_sec": response.response_time_sec,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
                "model_name": response.model_name,
            }
            
        except Exception as e:
            log.error(f"Query failed: {e}")
            raise
    
    def get_conversation_history(
        self,
        conversation_id: str,
    ) -> Dict[str, Any]:
        """Get conversation history."""
        if conversation_id not in self.pm.conversations:
            return {
                "conversation_id": conversation_id,
                "messages": [],
                "total_messages": 0,
            }
        
        conversation = self.pm.conversations[conversation_id]
        history = conversation["history"]
        
        messages = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
            }
            for msg in history
        ]
        
        return {
            "conversation_id": conversation_id,
            "messages": messages,
            "total_messages": len(messages),
        }
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history."""
        self.pm.clear_conversation(conversation_id)
        log.info(f"Conversation cleared: {conversation_id}")