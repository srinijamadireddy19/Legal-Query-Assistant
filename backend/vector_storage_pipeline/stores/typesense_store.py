"""
Typesense Vector Store
──────────────────────
Production-ready vector storage with Typesense.

Features:
✅ Collection creation with schema validation
✅ Batch indexing (not one-by-one)
✅ Embedding dimension check
✅ Logging indexing stats
✅ Duplicate ID handling
✅ Error recovery
✅ Version field (embedding_model_version)

Installation:
    pip install typesense
"""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.models import (
    CollectionSchema,
    CollectionInfo,
    IndexingResult,
    IndexingStatus,
    StorageConfig,
)

log = logging.getLogger(__name__)


class TypesenseVectorStore:
    """
    Vector store implementation using Typesense.
    Handles collection management, batch indexing, and search.
    """
    
    def __init__(self, config: StorageConfig):
        """
        Initialize Typesense client.
        
        Args:
            config: StorageConfig with connection details
        """
        self.config = config
        self.client = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Typesense server."""
        try:
            import typesense
            
            # Validate config
            config_errors = self.config.validate()
            if config_errors:
                raise ValueError(f"Invalid config: {', '.join(config_errors)}")
            
            # Create client
            self.client = typesense.Client({
                'nodes': [{
                    'host': self.config.host,
                    'port': self.config.port,
                    'protocol': self.config.protocol,
                }],
                'api_key': self.config.api_key,
                'connection_timeout_seconds': self.config.connection_timeout_seconds,
                'num_retries': self.config.num_retries,
            })
            
            # Test connection
            self.client.collections.retrieve()
            log.info(f"✓ Connected to Typesense at {self.config.get_connection_string()}")
            
        except ImportError:
            raise ImportError(
                "typesense package not installed. "
                "Install with: pip install typesense"
            )
        except Exception as e:
            log.error(f"Failed to connect to Typesense: {e}")
            raise
    
    # ══════════════════════════════════════════════════════════════════════
    # Collection Management
    # ══════════════════════════════════════════════════════════════════════
    
    def create_collection(
        self,
        schema: CollectionSchema,
        drop_if_exists: bool = False,
    ) -> bool:
        """
        Create a new collection with schema validation.
        
        Args:
            schema: CollectionSchema defining the collection structure
            drop_if_exists: If True, drop existing collection before creating
            
        Returns:
            True on success, False on failure
        """
        # Validate schema
        schema_errors = schema.validate()
        if schema_errors:
            log.error(f"Schema validation failed: {', '.join(schema_errors)}")
            return False
        
        try:
            # Check if collection exists
            exists = self.collection_exists(schema.name)
            
            if exists and drop_if_exists:
                log.warning(f"Dropping existing collection: {schema.name}")
                self.client.collections[schema.name].delete()
                exists = False
            
            if exists:
                log.warning(f"Collection '{schema.name}' already exists")
                
                # Verify schema matches
                existing = self.get_collection_info(schema.name)
                existing_dim = existing.get_embedding_dim()
                
                if existing_dim != schema.embedding_dim:
                    log.error(
                        f"Dimension mismatch: existing={existing_dim}, "
                        f"requested={schema.embedding_dim}"
                    )
                    return False
                
                log.info(f"Using existing collection: {schema.name}")
                return True
            
            # Create collection
            typesense_schema = schema.to_typesense_schema()
            self.client.collections.create(typesense_schema)
            
            log.info(
                f"✓ Created collection '{schema.name}' "
                f"(dim={schema.embedding_dim})"
            )
            return True
            
        except Exception as e:
            log.error(f"Failed to create collection: {e}")
            return False
    
    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        try:
            self.client.collections[name].retrieve()
            return True
        except Exception:
            return False
    
    def get_collection_info(self, name: str) -> Optional[CollectionInfo]:
        """Get information about a collection."""
        try:
            coll = self.client.collections[name].retrieve()
            return CollectionInfo(
                name=coll['name'],
                num_documents=coll.get('num_documents', 0),
                fields=coll.get('fields', []),
                created_at=coll.get('created_at'),
            )
        except Exception as e:
            log.error(f"Failed to retrieve collection info: {e}")
            return None
    
    def list_collections(self) -> List[CollectionInfo]:
        """List all collections."""
        try:
            collections = self.client.collections.retrieve()
            return [
                CollectionInfo(
                    name=c['name'],
                    num_documents=c.get('num_documents', 0),
                    fields=c.get('fields', []),
                    created_at=c.get('created_at'),
                )
                for c in collections
            ]
        except Exception as e:
            log.error(f"Failed to list collections: {e}")
            return []
    
    def drop_collection(self, name: str) -> bool:
        """Drop a collection."""
        try:
            self.client.collections[name].delete()
            log.info(f"✓ Dropped collection: {name}")
            return True
        except Exception as e:
            log.error(f"Failed to drop collection: {e}")
            return False
    
    # ══════════════════════════════════════════════════════════════════════
    # Batch Indexing
    # ══════════════════════════════════════════════════════════════════════
    
    def index_embeddings(
        self,
        collection_name: str,
        embeddings: List[Any],  # List of ChunkEmbedding objects
        embedding_model_version: Optional[str] = None,
    ) -> IndexingResult:
        """
        Batch index embeddings into Typesense.
        
        Args:
            collection_name: Name of the collection
            embeddings: List of ChunkEmbedding objects
            embedding_model_version: Optional version tag for the model
            
        Returns:
            IndexingResult with detailed statistics
        """
        start_time = time.time()
        
        result = IndexingResult(
            collection_name=collection_name,
            status=IndexingStatus.SUCCESS,
            total_documents=len(embeddings),
        )
        
        if not embeddings:
            result.warnings.append("No embeddings to index")
            return result
        
        # Get collection info for validation
        coll_info = self.get_collection_info(collection_name)
        if not coll_info:
            result.status = IndexingStatus.FAILED
            result.errors.append(f"Collection '{collection_name}' does not exist")
            return result
        
        # Validate embedding dimensions
        expected_dim = coll_info.get_embedding_dim()
        if expected_dim and embeddings[0].embedding_dim != expected_dim:
            result.status = IndexingStatus.FAILED
            result.errors.append(
                f"Dimension mismatch: collection expects {expected_dim}d, "
                f"got {embeddings[0].embedding_dim}d"
            )
            return result
        
        result.embedding_dim = embeddings[0].embedding_dim
        result.embedding_model = embeddings[0].model_name
        
        # Prepare documents in batches
        batches = self._batch_documents(embeddings, embedding_model_version)
        
        log.info(
            f"Indexing {len(embeddings)} documents in "
            f"{len(batches)} batches of {self.config.batch_size}"
        )
        
        # Index each batch
        error_count = 0
        
        for batch_idx, batch in enumerate(batches):
            try:
                batch_result = self._index_batch(
                    collection_name, batch, result
                )
                
                error_count += len(batch_result.get('failed', []))
                
                # Stop if too many errors
                if error_count >= self.config.max_errors:
                    result.errors.append(
                        f"Stopped after {error_count} errors "
                        f"(max_errors={self.config.max_errors})"
                    )
                    result.status = IndexingStatus.FAILED
                    break
                
                if (batch_idx + 1) % 10 == 0:
                    log.info(f"Processed {(batch_idx + 1) * self.config.batch_size} documents")
                    
            except Exception as e:
                log.error(f"Batch {batch_idx} failed: {e}")
                result.errors.append(f"Batch {batch_idx}: {str(e)}")
                
                if not self.config.skip_on_error:
                    result.status = IndexingStatus.FAILED
                    break
        
        # Finalize result
        result.total_time_sec = time.time() - start_time
        result.__post_init__()  # Recalculate derived stats
        
        # Determine final status
        if result.failed == 0:
            result.status = IndexingStatus.SUCCESS
        elif result.successful > 0:
            result.status = IndexingStatus.PARTIAL
        else:
            result.status = IndexingStatus.FAILED
        
        log.info(result.summary())
        
        return result
    
    def _batch_documents(
        self,
        embeddings: List[Any],
        embedding_model_version: Optional[str],
    ) -> List[List[Dict[str, Any]]]:
        """Convert embeddings to Typesense documents in batches."""
        batches = []
        current_batch = []
        existing_ids = set()
        
        for emb in embeddings:
            # Check for duplicates
            if self.config.check_duplicates:
                if emb.chunk_id in existing_ids:
                    log.warning(f"Duplicate ID skipped: {emb.chunk_id}")
                    continue
                existing_ids.add(emb.chunk_id)
            
            # Validate embedding
            if self.config.validate_embeddings:
                if not emb.embedding or len(emb.embedding) == 0:
                    log.warning(f"Empty embedding for {emb.chunk_id}, skipping")
                    continue
            
            # Create document
            doc = {
                "id": emb.chunk_id,
                "embedding": emb.embedding,
                "text": emb.text,
                "chunk_id": emb.chunk_id,
                "source_file": emb.source_file or "",
                "hierarchy": emb.hierarchy or [],
                "page_numbers": emb.page_numbers or [],
                "embedding_model": emb.model_name,
                "indexed_at": int(datetime.now().timestamp()),
                "metadata": emb.metadata or {},
            }
            
            # Add version if provided
            if embedding_model_version:
                doc["embedding_model_version"] = embedding_model_version
            
            current_batch.append(doc)
            
            # Flush batch when full
            if len(current_batch) >= self.config.batch_size:
                batches.append(current_batch)
                current_batch = []
        
        # Add remaining documents
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _index_batch(
        self,
        collection_name: str,
        batch: List[Dict[str, Any]],
        result: IndexingResult,
    ) -> Dict[str, Any]:
        """Index a single batch of documents."""
        try:
            # Import documents in batch
            # action=upsert: insert new or update existing
            response = self.client.collections[collection_name].documents.import_(
                batch,
                {'action': 'upsert'}
            )
            
            # Parse response (one result per document)
            for idx, doc_result in enumerate(response):
                if doc_result.get('success'):
                    result.successful += 1
                else:
                    result.failed += 1
                    error_msg = doc_result.get('error', 'Unknown error')
                    doc_id = batch[idx].get('id', f'doc_{idx}')
                    result.failed_ids.append(doc_id)
                    result.errors.append(f"{doc_id}: {error_msg}")
            
            return {'success': result.successful, 'failed': result.failed_ids}
            
        except Exception as e:
            # Batch-level error
            result.failed += len(batch)
            for doc in batch:
                result.failed_ids.append(doc.get('id', 'unknown'))
            raise
    
    # ══════════════════════════════════════════════════════════════════════
    # Search
    # ══════════════════════════════════════════════════════════════════════
    
    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        k: int = 10,
        filter_by: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search using vector similarity.
        
        Args:
            collection_name: Collection to search
            query_embedding: Query vector
            k: Number of results to return
            filter_by: Optional Typesense filter expression
            
        Returns:
            List of search results with scores
        """
        try:
            search_params = {
                "collection": collection_name,
                "q": "*",
                "vector_query": f"embedding:([{','.join(map(str, query_embedding))}], k:{k})",
                "exclude_fields": "embedding",  # Don't return embeddings in results
            }
            
            if filter_by:
                search_params["filter_by"] = filter_by
            
            # Use multi_search to avoid the 4000-char query string limit
            # that applies to the regular /documents/search endpoint
            response = self.client.multi_search.perform(
                {"searches": [search_params]},
                {}
            )
            results = response["results"][0]
            
            # Extract hits
            hits = results.get('hits', [])
            
            return [
                {
                    "id": hit['document']['id'],
                    "text": hit['document']['text'],
                    "score": hit.get('vector_distance', 0),
                    "hierarchy": hit['document'].get('hierarchy', []),
                    "source_file": hit['document'].get('source_file', ''),
                    "metadata": hit['document'].get('metadata', {}),
                }
                for hit in hits
            ]
            
        except Exception as e:
            log.error(f"Search failed: {e}")
            return []
    
    # ══════════════════════════════════════════════════════════════════════
    # Statistics
    # ══════════════════════════════════════════════════════════════════════
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics about a collection."""
        try:
            info = self.get_collection_info(collection_name)
            if not info:
                return {}
            
            return {
                "name": info.name,
                "num_documents": info.num_documents,
                "embedding_dim": info.get_embedding_dim(),
                "fields": len(info.fields),
            }
        except Exception as e:
            log.error(f"Failed to get stats: {e}")
            return {}
