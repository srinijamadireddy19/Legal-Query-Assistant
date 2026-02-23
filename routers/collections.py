"""
Collections Router
──────────────────
Endpoints for managing vector collections.
"""

from fastapi import APIRouter, HTTPException, Depends

from schemas.models import (
    CollectionCreateRequest,
    CollectionCreateResponse,
    CollectionListResponse,
    CollectionInfo,
)
from config.settings import get_pipeline_manager, PipelineManager

router = APIRouter(prefix="/collections", tags=["Collections"])


@router.post("/", response_model=CollectionCreateResponse)
async def create_collection(
    request: CollectionCreateRequest,
    pm: PipelineManager = Depends(get_pipeline_manager),
):
    """
    Create a new vector collection.
    
    Example:
    ```json
    {
        "name": "legal_docs",
        "embedding_dim": 1024,
        "model_name": "BAAI/bge-large-en-v1.5",
        "drop_if_exists": false
    }
    ```
    """
    try:
        success = pm.storage.create_collection_for_model(
            name=request.name,
            embedding_dim=request.embedding_dim,
            model_name=request.model_name,
            drop_if_exists=request.drop_if_exists,
        )
        
        if success:
            return CollectionCreateResponse(
                name=request.name,
                embedding_dim=request.embedding_dim,
                model_name=request.model_name,
                created=True,
                message=f"Collection '{request.name}' created successfully",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to create collection '{request.name}'",
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=CollectionListResponse)
async def list_collections(
    pm: PipelineManager = Depends(get_pipeline_manager),
):
    """
    List all collections.
    """
    try:
        collections = pm.storage.list_collections()
        
        collection_infos = [
            CollectionInfo(
                name=c.name,
                num_documents=c.num_documents,
                embedding_dim=c.get_embedding_dim() or 0,
            )
            for c in collections
        ]
        
        return CollectionListResponse(
            collections=collection_infos,
            total=len(collection_infos),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{collection_name}", response_model=CollectionInfo)
async def get_collection(
    collection_name: str,
    pm: PipelineManager = Depends(get_pipeline_manager),
):
    """
    Get information about a specific collection.
    """
    try:
        collection = pm.storage.get_collection_info(collection_name)
        
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found",
            )
        
        return CollectionInfo(
            name=collection.name,
            num_documents=collection.num_documents,
            embedding_dim=collection.get_embedding_dim() or 0,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{collection_name}")
async def delete_collection(
    collection_name: str,
    pm: PipelineManager = Depends(get_pipeline_manager),
):
    """
    Delete a collection.
    """
    try:
        success = pm.storage.drop_collection(collection_name)
        
        if success:
            return {
                "message": f"Collection '{collection_name}' deleted successfully",
                "collection_name": collection_name,
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found",
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))