from legal_rag_chunker import HierarchicalChunkingPipeline
import PyPDF2

file_path = "data/Partnership deed.pdf"

with open(file_path, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

pipeline = HierarchicalChunkingPipeline()

result = pipeline.process(text)

print(f"\nDetected Document Type: {result.detection_result.doc_type.value}")
print(f"Detection Confidence: {result.detection_result.confidence:.2f}")
print(f"\nTotal Chunks: {result.statistics['total_chunks']}")
print(f"Average Chunk Size: {result.statistics['avg_chunk_size']:.0f} characters")
print(f"Max Hierarchy Depth: {result.statistics['max_hierarchy_depth']}")
    
print("\n" + "-" * 80)
print("CHUNKS:")
print("-" * 80)
    
for i, chunk in enumerate(result.chunks[:5], 1):  # Show first 5 chunks
    print(f"\n[Chunk {i}] {chunk.chunk_id}")
    print(f"Hierarchy: {' > '.join(chunk.hierarchy)}")
    print(f"Content ({len(chunk.content)} chars):")
    print(chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content)
    
if len(result.chunks) > 5:
    print(f"\n... and {len(result.chunks) - 5} more chunks")