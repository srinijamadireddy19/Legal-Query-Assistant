"""
Complete Pipeline Demo: Ingestion → Chunking → Embedding
──────────────────────────────────────────────────────────
Shows the full workflow from raw documents to vector embeddings.

NOTE: This demo uses mock embeddings since fastembed isn't installed.
      In production, install: pip install fastembed
"""

import sys
sys.path.insert(0, "/home/claude")

import json
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════
# PART 1: Mock Embedder (for demo purposes - replace with real FastEmbed)
# ══════════════════════════════════════════════════════════════════════════

class MockEmbedder:
    """Mock embedder that generates fake embeddings for demo."""
    
    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension
        self.model_info_obj = type('obj', (object,), {
            'name': model_name,
            'dimension': dimension,
            'max_seq_length': 512,
            'description': f'Mock {model_name}',
        })()
    
    def load_model(self) -> bool:
        print(f"[MOCK] Loading {self.model_name} (dim={self.dimension})")
        return True
    
    def is_loaded(self) -> bool:
        return True
    
    @property
    def model_info(self):
        return self.model_info_obj
    
    def embed(self, texts):
        """Generate fake embeddings (hash-based for consistency)."""
        import hashlib
        embeddings = []
        for text in texts:
            # Use hash to generate deterministic fake embedding
            h = int(hashlib.md5(text.encode()).hexdigest(), 16)
            embedding = [(h >> (i * 8)) % 256 / 255.0 for i in range(self.dimension)]
            # Normalize
            norm = sum(x*x for x in embedding) ** 0.5
            embedding = [x/norm for x in embedding]
            embeddings.append(embedding)
        return embeddings
    
    def embed_single(self, text):
        return self.embed([text])[0]


# ══════════════════════════════════════════════════════════════════════════
# PART 2: Main Demo
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "END-TO-END PIPELINE DEMONSTRATION" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Document Ingestion
    # ─────────────────────────────────────────────────────────────────────
    
    print("\n" + "═" * 80)
    print("STEP 1: Document Ingestion (PDF / DOCX / Image → Text)")
    print("═" * 80)
    
    from doc_extracter import DocumentIngestionPipeline
    
    ingest = DocumentIngestionPipeline(ocr_dpi=300)
    
    # Ingest multiple documents
    files = [
        "data\demo\demo_native.pdf",
        "data\demo\demo_scanned.pdf",
        "data\demo\demo_doc.docx",
    ]
    
    ingestion_results = []
    for fpath in files:
        if Path(fpath).exists():
            result = ingest.ingest(fpath)
            ingestion_results.append(result)
            print(f"\n✓ {Path(fpath).name}")
            print(f"  Pages: {result.total_pages}, "
                  f"Native: {result.native_page_count}, "
                  f"OCR: {result.ocr_page_count}")
            print(f"  Text length: {len(result.plain_text)} chars")
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Hierarchical Chunking
    # ─────────────────────────────────────────────────────────────────────
    
    print("\n" + "═" * 80)
    print("STEP 2: Hierarchical Chunking (Structure-Aware)")
    print("═" * 80)
    
    from legal_rag_chunker import HierarchicalChunkingPipeline
    
    chunker = HierarchicalChunkingPipeline()
    
    all_chunks_by_doc = []
    
    for ing_result in ingestion_results:
        chunk_result = chunker.process(ing_result.plain_text)
        all_chunks_by_doc.append({
            'source': ing_result.file_path,
            'chunks': chunk_result.chunks,
            'detection': chunk_result.detection_result,
            'stats': chunk_result.statistics,
        })
        
        print(f"\n✓ {Path(ing_result.file_path).name}")
        print(f"  Document type: {chunk_result.detection_result.doc_type.value}")
        print(f"  Chunks: {len(chunk_result.chunks)}")
        print(f"  Avg size: {chunk_result.statistics['avg_chunk_size']:.0f} chars")
        print(f"  Max depth: {chunk_result.statistics['max_hierarchy_depth']}")
    
    # Show sample chunks
    print("\n" + "─" * 80)
    print("Sample Chunks:")
    print("─" * 80)
    for i, doc_data in enumerate(all_chunks_by_doc[:2], 1):
        print(f"\nDocument {i}: {Path(doc_data['source']).name}")
        for chunk in doc_data['chunks'][:2]:
            hier = " > ".join(chunk.hierarchy) if chunk.hierarchy else "(no hierarchy)"
            print(f"  [{chunk.chunk_id}] {hier}")
            print(f"    {chunk.content[:120]}...")
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Embedding Generation
    # ─────────────────────────────────────────────────────────────────────
    
    print("\n" + "═" * 80)
    print("STEP 3: Embedding Generation (BAAI/bge-large-en-v1.5)")
    print("═" * 80)
    print("\n[NOTE: Using mock embeddings for demo. Install fastembed for real embeddings.]")
    
    from embedding_pipeline.core.models import (
        ChunkEmbedding, 
        EmbeddingResult, 
        EmbeddingConfig,
        EmbeddingStatus
    )
    import time
    
    # Create config
    config = EmbeddingConfig(
        model_name="BAAI/bge-large-en-v1.5",
        fallback_model="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=16,
        normalize=True,
    )
    
    print(f"\nConfiguration:")
    print(f"  Primary model: {config.model_name}")
    print(f"  Fallback model: {config.fallback_model}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Normalize: {config.normalize}")
    
    # Create mock embedder (replace with real EmbeddingPipeline when fastembed is installed)
    embedder = MockEmbedder("BAAI/bge-large-en-v1.5", dimension=1024)
    embedder.load_model()
    
    # Embed all chunks
    all_embeddings = []
    
    for doc_data in all_chunks_by_doc:
        start_time = time.time()
        
        chunks = doc_data['chunks']
        texts = [c.content for c in chunks]
        
        # Generate embeddings
        embeddings = embedder.embed(texts)
        
        # Package as ChunkEmbedding objects
        chunk_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_emb = ChunkEmbedding(
                chunk_id=chunk.chunk_id,
                text=chunk.content,
                embedding=embedding,
                metadata=chunk.metadata,
                model_name=embedder.model_name,
                hierarchy=chunk.hierarchy,
                source_file=doc_data['source'],
            )
            chunk_embeddings.append(chunk_emb)
        
        elapsed = time.time() - start_time
        
        # Create result
        result = EmbeddingResult(
            embeddings=chunk_embeddings,
            status=EmbeddingStatus.SUCCESS,
            total_time_sec=elapsed,
            avg_time_per_chunk=elapsed / len(chunks),
        )
        
        all_embeddings.append({
            'source': doc_data['source'],
            'result': result,
        })
        
        print(f"\n✓ {Path(doc_data['source']).name}")
        print(f"  Chunks embedded: {len(chunk_embeddings)}")
        print(f"  Embedding dim: {chunk_embeddings[0].embedding_dim}")
        print(f"  Time: {elapsed:.2f}s ({elapsed/len(chunks)*1000:.1f}ms per chunk)")
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Summary & Export
    # ─────────────────────────────────────────────────────────────────────
    
    print("\n" + "═" * 80)
    print("STEP 4: Summary & Export")
    print("═" * 80)
    
    total_docs = len(all_embeddings)
    total_chunks = sum(len(e['result'].embeddings) for e in all_embeddings)
    total_vectors = sum(len(ce.embedding) * len(e['result'].embeddings) 
                       for e in all_embeddings 
                       for ce in e['result'].embeddings)
    
    print(f"\n📊 Pipeline Statistics:")
    print(f"  Documents processed: {total_docs}")
    print(f"  Chunks created: {total_chunks}")
    print(f"  Vectors generated: {total_chunks}")
    print(f"  Total float values: {total_vectors:,}")
    print(f"  Memory estimate: {total_vectors * 4 / 1024 / 1024:.2f} MB (fp32)")
    
    # Export sample to JSON
    sample_export = []
    for emb_data in all_embeddings[:1]:
        for chunk_emb in emb_data['result'].embeddings[:3]:
            sample_export.append({
                'chunk_id': chunk_emb.chunk_id,
                'source': Path(chunk_emb.source_file).name,
                'hierarchy': chunk_emb.hierarchy,
                'text_preview': chunk_emb.text[:100],
                'embedding_dim': chunk_emb.embedding_dim,
                'embedding_sample': chunk_emb.embedding[:5],  # First 5 values
            })
    
    print(f"\n📄 Sample Export (JSON):")
    print(json.dumps(sample_export, indent=2))
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Vector DB Integration Points
    # ─────────────────────────────────────────────────────────────────────
    
    print("\n" + "═" * 80)
    print("STEP 5: Vector DB Integration (Pseudocode)")
    print("═" * 80)
    
    print("""
## ChromaDB Example:
    
    import chromadb
    client = chromadb.Client()
    collection = client.create_collection("legal_docs")
    
    for emb_data in all_embeddings:
        for chunk_emb in emb_data['result'].embeddings:
            collection.add(
                ids=[chunk_emb.chunk_id],
                embeddings=[chunk_emb.embedding],
                metadatas=[{
                    'source': chunk_emb.source_file,
                    'hierarchy': ' > '.join(chunk_emb.hierarchy),
                    **chunk_emb.metadata,
                }],
                documents=[chunk_emb.text],
            )

## Pinecone Example:
    
    import pinecone
    index = pinecone.Index("legal-docs")
    
    vectors = []
    for emb_data in all_embeddings:
        for chunk_emb in emb_data['result'].embeddings:
            vectors.append({
                'id': chunk_emb.chunk_id,
                'values': chunk_emb.embedding,
                'metadata': {
                    'text': chunk_emb.text[:1000],
                    'source': chunk_emb.source_file,
                    'hierarchy': ' > '.join(chunk_emb.hierarchy),
                }
            })
    
    index.upsert(vectors=vectors)

## Query Example:
    
    query = "What are the payment terms?"
    query_embedding = embedder.embed_single(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    for result in results:
        print(f"Hierarchy: {result['metadata']['hierarchy']}")
        print(f"Text: {result['document']}")
""")
    
    print("\n" + "═" * 80)
    print("✅ Demo Complete!")
    print("═" * 80)
    print("\nNext steps:")
    print("  1. Install: pip install fastembed")
    print("  2. Replace MockEmbedder with real EmbeddingPipeline")
    print("  3. Connect to your vector database")
    print("  4. Start querying!")


if __name__ == "__main__":
    main()