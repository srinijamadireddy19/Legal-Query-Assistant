import os
from dotenv import load_dotenv
from vector_storage_pipeline.pipeline import VectorStoragePipeline
from vector_storage_pipeline.core.models import StorageConfig
from llm_pipeline.pipeline import LLMPipeline
from llm_pipeline.providers.groq_provider import GroqModels
load_dotenv()

TYPESENSE_API_KEY = os.getenv("TYPESENSE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Stage 1: Ingest any document format
from doc_extracter import DocumentIngestionPipeline
ingest = DocumentIngestionPipeline()
doc = ingest.ingest("data/partnership deed.pdf")

print("Ingestion done")

# Stage 2: Hierarchical chunking
from legal_rag_chunker import HierarchicalChunkingPipeline
chunker = HierarchicalChunkingPipeline()
chunks = chunker.process(doc.plain_text)

print("Chunking done")

# Stage 3: Generate embeddings
from embedding_pipeline import EmbeddingPipeline, EmbeddingConfig
embedder = EmbeddingPipeline(EmbeddingConfig(
    model_name="BAAI/bge-base-en-v1.5",
    fallback_model="sentence-transformers/all-MiniLM-L6-v2",
))
print("model loaded")
emb_result = embedder.embed_chunks(chunks.chunks)
print("Embedding done")

# Stage 4: Store in Typesense (NEW!)

storage = VectorStoragePipeline(StorageConfig(
    host="msyjt95x41fv2ckqp-1.a2.typesense.net",
    port=443,
    api_key=TYPESENSE_API_KEY,
    protocol="https",
))

# Create (or recreate) collection — drop_if_exists=True replaces any stale
# collection whose embedding dimension doesn't match the current model
storage.create_collection_for_model(
    name="legal_docs",
    embedding_dim=embedder.model_info.dimension,
    model_name="BAAI/bge-base-en-v1.5",
    drop_if_exists=True,
)

print("Collection created")

# Index with batch processing + stats + dedup
result = storage.index(
    collection_name="legal_docs",
    embeddings=emb_result.embeddings,
    model_version="v1.0",
)

print(result.summary())

print("Indexing done")

llm_pipeline = LLMPipeline(
    GroqModels.llama_8b_instant(GROQ_API_KEY),
    embedder,
    storage
)

r1 = llm_pipeline.query("What are the payment terms?", "legal_docs")
print(r1.answer)
print(f"Type: {r1.query_type.value}")  # NEW
print(f"Quality: {r1.context_quality.value}")  # HIGH/MEDIUM/LOW/NO_CONTEXT
print(f"Citations: {r1.citations}")

# Follow-up (automatically detected!)
r2 = llm_pipeline.query("What about late fees?", "legal_docs")
print(r2.answer)
print(f"Type: {r2.query_type.value}")  # FOLLOWUP

# No context found (score < 0.3)
r3 = llm_pipeline.query("What is the vacation policy?", "legal_docs")
print(r3.answer)
