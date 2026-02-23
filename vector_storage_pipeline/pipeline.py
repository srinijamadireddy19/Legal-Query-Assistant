"""
Vector Storage Pipeline
───────────────────────
High-level API for storing and retrieving embeddings.
Abstracts Typesense details behind a clean interface.

Usage:
    from vector_storage import VectorStoragePipeline, StorageConfig
    
    config = StorageConfig(
        host="localhost",
        port=8108,
        api_key="xyz",
    )
    
    storage = VectorStoragePipeline(config)
    
    # Create collection
    storage.create_collection_for_model(
        name="legal_docs",
        embedding_dim=1024,
        model_name="BAAI/bge-large-en-v1.5",
    )
    
    # Index embeddings
    result = storage.index(
        collection_name="legal_docs",
        embeddings=chunk_embeddings,
        model_version="v1.0",
    )
"""

import logging
from typing import List, Dict, Any, Optional

from .stores.typesense_store import TypesenseVectorStore
from .core.models import (
    CollectionSchema,
    CollectionInfo,
    IndexingResult,
    StorageConfig,
)

log = logging.getLogger(__name__)


class VectorStoragePipeline:
    """
    High-level vector storage pipeline.
    Simplifies common operations like collection creation and batch indexing.
    """
    
    def __init__(self, config: StorageConfig):
        """
        Initialize storage pipeline.
        
        Args:
            config: StorageConfig with Typesense connection details
        """
        self.config = config
        self.store = TypesenseVectorStore(config)
        
        log.info(
            f"Vector storage pipeline initialized "
            f"({config.get_connection_string()})"
        )
    
    # ══════════════════════════════════════════════════════════════════════
    # Collection Management (Simplified API)
    # ══════════════════════════════════════════════════════════════════════
    
    def create_collection_for_model(
        self,
        name: str,
        embedding_dim: int,
        model_name: str = "",
        drop_if_exists: bool = False,
        additional_fields: List[Dict[str, Any]] = None,
    ) -> bool:
        """
        Create a collection optimized for a specific embedding model.
        
        Args:
            name: Collection name
            embedding_dim: Dimension of embeddings (384, 768, 1024, etc.)
            model_name: Name of the embedding model (for documentation)
            drop_if_exists: Drop existing collection if it exists
            additional_fields: Custom fields beyond the defaults
            
        Returns:
            True on success
        """
        schema = CollectionSchema(
            name=name,
            embedding_dim=embedding_dim,
            fields=additional_fields or [],
        )
        
        success = self.store.create_collection(schema, drop_if_exists)
        
        if success and model_name:
            log.info(f"Collection '{name}' ready for {model_name} ({embedding_dim}d)")
        
        return success
    
    def create_collection(
        self,
        schema: CollectionSchema,
        drop_if_exists: bool = False,
    ) -> bool:
        """
        Create a collection with custom schema.
        
        Args:
            schema: CollectionSchema defining structure
            drop_if_exists: Drop existing collection
            
        Returns:
            True on success
        """
        return self.store.create_collection(schema, drop_if_exists)
    
    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        return self.store.collection_exists(name)
    
    def get_collection_info(self, name: str) -> Optional[CollectionInfo]:
        """Get information about a collection."""
        return self.store.get_collection_info(name)
    
    def list_collections(self) -> List[CollectionInfo]:
        """List all collections."""
        return self.store.list_collections()
    
    def drop_collection(self, name: str) -> bool:
        """Drop a collection."""
        return self.store.drop_collection(name)
    
    # ══════════════════════════════════════════════════════════════════════
    # Indexing (Simplified API)
    # ══════════════════════════════════════════════════════════════════════
    
    def index(
        self,
        collection_name: str,
        embeddings: List[Any],
        model_version: Optional[str] = None,
    ) -> IndexingResult:
        """
        Index embeddings into a collection.
        
        Args:
            collection_name: Target collection
            embeddings: List of ChunkEmbedding objects
            model_version: Optional version tag (e.g., "v1.0", "2024-02-01")
            
        Returns:
            IndexingResult with detailed statistics
        """
        if not embeddings:
            log.warning("No embeddings to index")
            result = IndexingResult(
                collection_name=collection_name,
                status=IndexingResult.SUCCESS,
            )
            result.warnings.append("No embeddings provided")
            return result
        
        # Validate collection exists
        if not self.collection_exists(collection_name):
            log.error(f"Collection '{collection_name}' does not exist")
            result = IndexingResult(
                collection_name=collection_name,
                status=IndexingResult.FAILED,
                total_documents=len(embeddings),
            )
            result.errors.append("Collection does not exist")
            return result
        
        # Index
        log.info(
            f"Indexing {len(embeddings)} embeddings into '{collection_name}'"
        )
        
        return self.store.index_embeddings(
            collection_name=collection_name,
            embeddings=embeddings,
            embedding_model_version=model_version,
        )
    
    def index_from_embedding_result(
        self,
        collection_name: str,
        embedding_result: Any,  # EmbeddingResult object
        model_version: Optional[str] = None,
    ) -> IndexingResult:
        """
        Index embeddings directly from EmbeddingPipeline result.
        
        Args:
            collection_name: Target collection
            embedding_result: EmbeddingResult from embedding pipeline
            model_version: Optional version tag
            
        Returns:
            IndexingResult
        """
        return self.index(
            collection_name=collection_name,
            embeddings=embedding_result.embeddings,
            model_version=model_version,
        )
    
    # ══════════════════════════════════════════════════════════════════════
    # Search (Simplified API)
    # ══════════════════════════════════════════════════════════════════════
    
    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        k: int = 10,
        filter_by: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search.
        
        Args:
            collection_name: Collection to search
            query_embedding: Query vector
            k: Number of results
            filter_by: Optional filter (e.g., "source_file:=contract.pdf")
            
        Returns:
            List of results with text, score, hierarchy, etc.
        """
        return self.store.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            k=k,
            filter_by=filter_by,
        )
    
    def search_by_text(
        self,
        collection_name: str,
        query_text: str,
        embedder: Any,  # EmbeddingPipeline instance
        k: int = 10,
        filter_by: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search using text query (auto-embeds query).
        
        Args:
            collection_name: Collection to search
            query_text: Text query
            embedder: EmbeddingPipeline instance to embed query
            k: Number of results
            filter_by: Optional filter
            
        Returns:
            List of results
        """
        # Embed query
        query_embedding = embedder.embedder.embed_single(query_text)
        
        if not query_embedding:
            log.error("Failed to embed query")
            return []
        
        return self.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            k=k,
            filter_by=filter_by,
        )
    
    # ══════════════════════════════════════════════════════════════════════
    # Statistics
    # ══════════════════════════════════════════════════════════════════════
    
    def get_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection."""
        return self.store.get_collection_stats(collection_name)
    
    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all collections."""
        collections = self.list_collections()
        return [
            {
                "name": c.name,
                "num_documents": c.num_documents,
                "embedding_dim": c.get_embedding_dim(),
            }
            for c in collections
        ]
    
    # ══════════════════════════════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════════════════════════════
    
    def ping(self) -> bool:
        """Test connection to Typesense."""
        try:
            self.store.client.collections.retrieve()
            return True
        except Exception:
            return False
    
    def __repr__(self) -> str:
        return (
            f"VectorStoragePipeline("
            f"{self.config.get_connection_string()}, "
            f"batch_size={self.config.batch_size})"
        )
