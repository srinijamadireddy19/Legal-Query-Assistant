"""
Query Router
────────────
Endpoints for RAG queries and conversation management.
"""

from fastapi import APIRouter, HTTPException, Depends

from schemas.models import (
    QueryRequest,
    QueryResponse,
    ConversationHistoryResponse,
)
from services.query_service import QueryService
from config.settings import get_pipeline_manager, PipelineManager

router = APIRouter(prefix="/query", tags=["Query"])


def get_query_service(pm: PipelineManager = Depends(get_pipeline_manager)):
    """Dependency injection for QueryService."""
    return QueryService(pm)


@router.post("/", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
):
    """
    Query the RAG system with a question.
    
    Features:
    - Automatic query classification (new/followup/clarification)
    - Context-aware prompts
    - Citation generation
    - Multi-turn conversation support
    
    Example:
    ```json
    {
        "query": "What are the payment terms?",
        "collection_name": "legal_docs",
        "conversation_id": null,  // Optional, for multi-turn
        "k": 5,                   // Number of contexts
        "filter_by": null         // Optional Typesense filter
    }
    ```
    """
    try:
        result = await query_service.query(
            query=request.query,
            collection_name=request.collection_name,
            conversation_id=request.conversation_id,
            k=request.k,
            filter_by=request.filter_by,
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    conversation_id: str,
    query_service: QueryService = Depends(get_query_service),
):
    """
    Get the history of a conversation.
    """
    try:
        history = query_service.get_conversation_history(conversation_id)
        return ConversationHistoryResponse(**history)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conversation_id}")
async def clear_conversation(
    conversation_id: str,
    query_service: QueryService = Depends(get_query_service),
):
    """
    Clear conversation history (start new conversation).
    """
    try:
        query_service.clear_conversation(conversation_id)
        return {"message": "Conversation cleared", "conversation_id": conversation_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))