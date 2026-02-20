from legal_rag_chunker import HierarchicalChunkingPipeline
from doc_extracter import DocumentIngestionPipeline
from embedding_pipeline import EmbeddingPipeline
from embedding_pipeline.core.models import (
    ChunkEmbedding,
    EmbeddingConfig,
    EmbeddingResult,
    EmbeddingStatus,
)
from embedding_pipeline.embedders.fastembed_embedder import FastEmbedder

file_path = "data/sample.pdf"

print('*'*60)
ingest_pipeline = DocumentIngestionPipeline()
ingest_result = ingest_pipeline.ingest(file_path)

print(ingest_result.summary())

text = ingest_result.plain_text

print('*'*60)
chunking_pipeline = HierarchicalChunkingPipeline()

chunking_result = chunking_pipeline.process(text)

print(f"\nDetected Document Type: {chunking_result.detection_result.doc_type.value}")
print(f"Detection Confidence: {chunking_result.detection_result.confidence:.2f}")
print(f"\nTotal Chunks: {chunking_result.statistics['total_chunks']}")
print(f"Average Chunk Size: {chunking_result.statistics['avg_chunk_size']:.0f} characters")
print(f"Max Hierarchy Depth: {chunking_result.statistics['max_hierarchy_depth']}")
    
print("\n" + "-" * 80)
print("CHUNKS:")
print("-" * 80)
    
for i, chunk in enumerate(chunking_result.chunks[:5], 1):  # Show first 5 chunks
    print(f"\n[Chunk {i}] {chunk.chunk_id}")
    print(f"Hierarchy: {' > '.join(chunk.hierarchy)}")
    print(f"Content ({len(chunk.content)} chars):")
    print(chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content)
    
if len(chunking_result.chunks) > 5:
    print(f"\n... and {len(chunking_result.chunks) - 5} more chunks")

print('*'*60)
config = EmbeddingConfig()
print(config.model_name)
print(config.fallback_model)

embedder = EmbeddingPipeline(config)


data = [chunk.content for chunk in chunking_result.chunks]

embeddings = embedder.embed_chunks(data)
chunk_emb = []
for i, chunk in enumerate(chunking_result.chunks):
    chunk_emb.append(ChunkEmbedding(
        text=chunk.content,
        chunk_id=chunk.chunk_id,
        embedding=embeddings[i],
    ))

result = EmbeddingResult(embeddings=chunk_emb,status=EmbeddingStatus.SUCCESS)
print(result.summary())