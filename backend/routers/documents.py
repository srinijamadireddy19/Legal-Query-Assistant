"""
Document Router
───────────────
Endpoints for document upload and processing.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List

from schemas.models import (
    DocumentUploadResponse,
    DocumentProcessRequest,
    DocumentProcessResponse,
    DocumentStatusResponse,
    DocumentStatus,
)
from services.document_service import DocumentService
from config.settings import get_pipeline_manager, PipelineManager

router = APIRouter(prefix="/documents", tags=["Documents"])


def get_document_service(pm: PipelineManager = Depends(get_pipeline_manager)):
    """Dependency injection for DocumentService."""
    return DocumentService(pm)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    doc_service: DocumentService = Depends(get_document_service),
):
    """
    Upload a document for processing.
    
    Supported formats: PDF, DOCX, DOC, PNG, JPG, TXT, MD
    """
    try:
        # Read file content
        content = await file.read()
        
        # Upload
        doc_id = await doc_service.upload_document(file.filename, content)
        
        return DocumentUploadResponse(
            document_id=doc_id,
            filename=file.filename,
            status=DocumentStatus.PENDING,
            message="Document uploaded successfully. Use /documents/{document_id}/process to start processing.",
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{document_id}/process", response_model=DocumentProcessResponse)
async def process_document(
    document_id: str,
    request: DocumentProcessRequest,
    doc_service: DocumentService = Depends(get_document_service),
):
    """
    Process an uploaded document through the complete pipeline:
    1. Ingestion (PDF/DOCX/Image → Text)
    2. Chunking (Structure-aware splitting)
    3. Embedding (Generate vectors)
    4. Indexing (Store in Typesense)
    """
    try:
        result = await doc_service.process_document(
            doc_id=document_id,
            collection_name=request.collection_name,
            model_version=request.model_version,
        )
        
        return DocumentProcessResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: str,
    doc_service: DocumentService = Depends(get_document_service),
):
    """
    Get the processing status of a document.
    """
    try:
        status = doc_service.get_document_status(document_id)
        return DocumentStatusResponse(**status)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))