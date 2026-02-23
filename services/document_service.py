"""
Document Processing Service
───────────────────────────
Handles document upload → ingestion → chunking → embedding → storage.
"""

import os
import uuid
import logging
import time
from pathlib import Path
from typing import Dict, Any

from config.settings import RAGConfig, PipelineManager

log = logging.getLogger(__name__)


class DocumentService:
    """
    Service for document processing.
    Orchestrates the full pipeline from upload to indexing.
    """
    
    def __init__(self, pipeline_manager: PipelineManager):
        self.pm = pipeline_manager
        
        # Ensure upload directory exists
        os.makedirs(RAGConfig.UPLOAD_DIR, exist_ok=True)
        
        # Document status tracking (in-memory for now)
        self.documents: Dict[str, Dict[str, Any]] = {}
    
    async def upload_document(self, filename: str, content: bytes) -> str:
        """
        Save uploaded document.
        
        Returns:
            document_id
        """
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Save file
        file_path = os.path.join(RAGConfig.UPLOAD_DIR, f"{doc_id}_{filename}")
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Track document
        self.documents[doc_id] = {
            "id": doc_id,
            "filename": filename,
            "file_path": file_path,
            "status": "pending",
            "progress": 0,
            "stage": "uploaded",
            "created_at": time.time(),
        }
        
        log.info(f"Document uploaded: {doc_id} ({filename})")
        
        return doc_id
    
    async def process_document(
        self,
        doc_id: str,
        collection_name: str,
        model_version: str = "v1.0",
    ) -> Dict[str, Any]:
        """
        Process document through the complete pipeline.
        
        Args:
            doc_id: Document ID from upload
            collection_name: Target collection
            model_version: Version tag for embeddings
            
        Returns:
            Processing result dict
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")
        
        doc = self.documents[doc_id]
        doc["status"] = "processing"
        
        start_time = time.time()
        errors = []
        
        try:
            # ── Stage 1: Ingestion ────────────────────────────────────────
            doc["stage"] = "ingestion"
            doc["progress"] = 20
            
            log.info(f"[{doc_id}] Stage 1: Ingesting document...")
            ingest_result = self.pm.ingestion.ingest(doc["file_path"])
            
            if ingest_result.status.value == "failed":
                raise Exception(f"Ingestion failed: {ingest_result.errors}")
            
            errors.extend(ingest_result.errors)
            
            log.info(
                f"[{doc_id}] Ingestion complete: {ingest_result.total_pages} pages, "
                f"{ingest_result.native_page_count} native, {ingest_result.ocr_page_count} OCR"
            )
            
            # ── Stage 2: Chunking ─────────────────────────────────────────
            doc["stage"] = "chunking"
            doc["progress"] = 40
            
            log.info(f"[{doc_id}] Stage 2: Chunking document...")
            chunk_result = self.pm.chunker.process(ingest_result.plain_text)
            
            log.info(
                f"[{doc_id}] Chunking complete: {len(chunk_result.chunks)} chunks, "
                f"type: {chunk_result.detection_result.doc_type.value}"
            )
            
            # ── Stage 3: Embedding ────────────────────────────────────────
            doc["stage"] = "embedding"
            doc["progress"] = 60
            
            log.info(f"[{doc_id}] Stage 3: Generating embeddings...")
            emb_result = self.pm.embedder.embed_chunks(
                chunks=chunk_result.chunks,
                source_file=doc["filename"],
            )
            
            if emb_result.status.value == "failed":
                raise Exception(f"Embedding failed: {emb_result.errors}")
            
            log.info(
                f"[{doc_id}] Embedding complete: {len(emb_result.embeddings)} vectors, "
                f"model: {emb_result.model_name}"
            )
            
            # ── Stage 4: Indexing ─────────────────────────────────────────
            doc["stage"] = "indexing"
            doc["progress"] = 80
            
            log.info(f"[{doc_id}] Stage 4: Indexing to '{collection_name}'...")
            index_result = self.pm.storage.index(
                collection_name=collection_name,
                embeddings=emb_result.embeddings,
                model_version=model_version,
            )
            
            if index_result.status.value == "failed":
                raise Exception(f"Indexing failed: {index_result.errors}")
            
            log.info(
                f"[{doc_id}] Indexing complete: {index_result.successful} docs indexed"
            )
            
            # ── Complete ──────────────────────────────────────────────────
            doc["status"] = "completed"
            doc["progress"] = 100
            doc["stage"] = "completed"
            
            processing_time = time.time() - start_time
            
            result = {
                "document_id": doc_id,
                "filename": doc["filename"],
                "status": "completed",
                "total_pages": ingest_result.total_pages,
                "native_pages": ingest_result.native_page_count,
                "ocr_pages": ingest_result.ocr_page_count,
                "total_chunks": len(chunk_result.chunks),
                "chunks_indexed": index_result.successful,
                "processing_time_sec": processing_time,
                "errors": errors + index_result.errors,
            }
            
            log.info(
                f"[{doc_id}] Processing complete in {processing_time:.2f}s: "
                f"{result['total_pages']}p → {result['total_chunks']}c → "
                f"{result['chunks_indexed']} indexed"
            )
            
            return result
            
        except Exception as e:
            doc["status"] = "failed"
            doc["stage"] = "failed"
            error_msg = str(e)
            errors.append(error_msg)
            
            log.error(f"[{doc_id}] Processing failed: {e}")
            
            return {
                "document_id": doc_id,
                "filename": doc["filename"],
                "status": "failed",
                "total_pages": 0,
                "native_pages": 0,
                "ocr_pages": 0,
                "total_chunks": 0,
                "chunks_indexed": 0,
                "processing_time_sec": time.time() - start_time,
                "errors": errors,
            }
    
    def get_document_status(self, doc_id: str) -> Dict[str, Any]:
        """Get document processing status."""
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")
        
        doc = self.documents[doc_id]
        
        return {
            "document_id": doc_id,
            "filename": doc["filename"],
            "status": doc["status"],
            "progress_percentage": doc.get("progress", 0),
            "current_stage": doc.get("stage", "unknown"),
            "message": self._get_status_message(doc),
        }
    
    def _get_status_message(self, doc: dict) -> str:
        """Generate status message."""
        status = doc["status"]
        stage = doc.get("stage", "")
        
        messages = {
            "pending": "Document uploaded, waiting to process",
            "processing": f"Processing document: {stage}...",
            "completed": "Document processed successfully",
            "failed": f"Processing failed at stage: {stage}",
        }
        
        return messages.get(status, "Unknown status")