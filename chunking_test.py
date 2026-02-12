from legal_rag_chunker import HierarchicalChunkingPipeline
from doc_extracter import DocumentIngestionPipeline

file_path = "data/sample.pdf"

ingest_pipeline = DocumentIngestionPipeline()
ingest_result = ingest_pipeline.ingest(file_path)

print(ingest_result.summary())

text = ingest_result.plain_text

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