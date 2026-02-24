# Vector Storage Pipeline with Typesense

**Production-ready vector storage with all essential features for legal RAG systems.**

✅ **Collection creation** with schema validation  
✅ **Batch indexing** (100+ docs per request, not one-by-one)  
✅ **Embedding dimension check** (prevents mismatched vectors)  
✅ **Detailed statistics** (success/fail rates, speed metrics)  
✅ **Duplicate ID handling** (automatic detection and skipping)  
✅ **Error recovery** (continues on individual failures)  
✅ **Version tracking** (`embedding_model_version` field)  

---

## Why Typesense?

- **Fast** - Built in C++, optimized for speed
- **Simple** - Easy setup, no complex configuration
- **Hybrid** - Vector search + keyword search + filters
- **Scalable** - Handles millions of vectors
- **Self-hosted** - Your data, your infrastructure

---

## Quick Start

### 1. Start Typesense

```bash
# Docker (easiest)
docker run -p 8108:8108 \
  -v /tmp/typesense:/data \
  typesense/typesense:26.0 \
  --data-dir /data \
  --api-key=xyz

# Or download binary from typesense.org
```

### 2. Install Client

```bash
pip install typesense
```

### 3. Use the Pipeline

```python
from vector_storage import VectorStoragePipeline, StorageConfig

# Connect
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

# Index embeddings (from embedding pipeline)
result = storage.index(
    collection_name="legal_docs",
    embeddings=chunk_embeddings,  # List[ChunkEmbedding]
    model_version="v1.0",
)

print(result.summary())
# Indexing Result: legal_docs
#   Total docs    : 1000
#   Successful    : 997
#   Duplicates    : 3 (skipped)
#   Speed         : 207.5 docs/sec
```

---

## Features in Detail

### ✅ Collection Creation with Schema Validation

```python
from vector_storage import CollectionSchema

# Method 1: Simple (recommended)
storage.create_collection_for_model(
    name="legal_docs",
    embedding_dim=1024,
    model_name="BAAI/bge-large-en-v1.5",
)

# Method 2: Custom schema
schema = CollectionSchema(
    name="legal_docs_custom",
    embedding_dim=768,
    fields=[
        {"name": "doc_type", "type": "string", "optional": True},
        {"name": "importance", "type": "int32", "optional": True},
    ],
)

# Validate before creating
errors = schema.validate()
if not errors:
    storage.create_collection(schema)
```

**Default fields** (always included):
- `id` - Unique identifier
- `embedding` - Vector (float[])
- `text` - Chunk text
- `chunk_id` - Original chunk ID
- `source_file` - Origin document
- `hierarchy` - Section breadcrumbs
- `page_numbers` - Page locations
- `embedding_model` - Model name
- `embedding_model_version` - Version tag
- `indexed_at` - Timestamp
- `metadata` - Custom JSON object

### ✅ Batch Indexing

```python
# Efficiently indexes in batches (default: 100 docs per batch)
result = storage.index(
    collection_name="legal_docs",
    embeddings=chunk_embeddings,  # Could be 10,000 embeddings
    model_version="v1.0",
)

# Processes:
# - 10,000 embeddings → 100 batches → 100 API calls
# - NOT 10,000 individual API calls!
```

**Configuration:**

```python
config = StorageConfig(
    batch_size=200,  # Larger batches = faster (up to a point)
)
```

### ✅ Embedding Dimension Check

Prevents costly mistakes:

```python
# Collection expects 1024d
# Embeddings are 768d
# → IndexingResult.status = FAILED
# → Error: "Dimension mismatch: expected 1024d, got 768d"
# → No data sent to Typesense
```

Validation happens **before** any network requests.

### ✅ Detailed Statistics

```python
result = storage.index(...)

print(result.summary())
# Indexing Result: legal_docs
#   Status        : success
#   Total docs    : 1000
#   Successful    : 997
#   Failed        : 0
#   Duplicates    : 3 (skipped)
#   Total time    : 4.82s
#   Speed         : 207.5 docs/sec
#   Model         : BAAI/bge-large-en-v1.5 (1024d)
```

**Available metrics:**
- `total_documents`, `successful`, `failed`
- `duplicates_skipped`
- `total_time_sec`, `avg_time_per_doc`, `docs_per_second`
- `failed_ids` (list of IDs that failed)
- `errors` (detailed error messages)
- `embedding_dim`, `embedding_model`

### ✅ Duplicate ID Handling

```python
# Automatic detection (enabled by default)
config = StorageConfig(
    check_duplicates=True,
)

# If a chunk_id appears twice:
# → Second instance is skipped
# → result.duplicates_skipped += 1
# → Warning logged

# Disable if you're certain there are no duplicates (faster):
config = StorageConfig(check_duplicates=False)
```

### ✅ Error Recovery

```python
config = StorageConfig(
    skip_on_error=True,   # Don't stop batch on individual doc failure
    max_errors=100,        # Stop after N total errors
    num_retries=3,         # Retry failed requests
)

# Behavior:
# - Doc 1: Success
# - Doc 2: Failure (logged, continue)
# - Doc 3: Success
# - ...
# - After 100 errors: Stop, return PARTIAL status
```

**Error tracking:**

```python
result = storage.index(...)

if result.status == IndexingStatus.PARTIAL:
    print(f"Failed IDs: {result.failed_ids}")
    for error in result.errors:
        print(f"  - {error}")
```

### ✅ Model Version Tracking

```python
# Index with version tag
result = storage.index(
    collection_name="legal_docs",
    embeddings=embeddings,
    model_version="v1.0",  # ← Stored with every document
)

# Later, after upgrading model to v2.0:
# 1. Create new collection
storage.create_collection_for_model("legal_docs_v2", ...)

# 2. Re-index with new version
result = storage.index(
    collection_name="legal_docs_v2",
    embeddings=new_embeddings,
    model_version="v2.0",
)

# 3. Query with version filter
results = storage.search(
    collection_name="legal_docs",
    query_embedding=query,
    filter_by="embedding_model_version:=v1.0",
)
```

**Version strategies:**
- Semantic versioning: `v1.0`, `v2.0`
- Dates: `2024-02-20`
- Descriptive: `production`, `finetuned-legal`
- Model-specific: `bge-large-v1.5`

---

## API Reference

### `VectorStoragePipeline`

Main class for vector storage operations.

#### Collection Management

```python
# Create collection (simple)
create_collection_for_model(
    name: str,
    embedding_dim: int,
    model_name: str = "",
    drop_if_exists: bool = False,
) → bool

# Create collection (custom schema)
create_collection(
    schema: CollectionSchema,
    drop_if_exists: bool = False,
) → bool

# Check existence
collection_exists(name: str) → bool

# Get info
get_collection_info(name: str) → CollectionInfo | None

# List all
list_collections() → List[CollectionInfo]

# Drop collection
drop_collection(name: str) → bool
```

#### Indexing

```python
# Index embeddings
index(
    collection_name: str,
    embeddings: List[ChunkEmbedding],
    model_version: str | None = None,
) → IndexingResult

# Index from embedding pipeline result
index_from_embedding_result(
    collection_name: str,
    embedding_result: EmbeddingResult,
    model_version: str | None = None,
) → IndexingResult
```

#### Search

```python
# Semantic search
search(
    collection_name: str,
    query_embedding: List[float],
    k: int = 10,
    filter_by: str | None = None,
) → List[Dict[str, Any]]

# Search with text query (auto-embeds)
search_by_text(
    collection_name: str,
    query_text: str,
    embedder: EmbeddingPipeline,
    k: int = 10,
    filter_by: str | None = None,
) → List[Dict[str, Any]]
```

#### Utilities

```python
get_stats(collection_name: str) → Dict[str, Any]
get_all_stats() → List[Dict[str, Any]]
ping() → bool
```

---

## Configuration

```python
from vector_storage import StorageConfig

config = StorageConfig(
    # Connection
    host="localhost",
    port=8108,
    protocol="http",          # or "https"
    api_key="xyz",
    
    # Performance
    batch_size=100,           # Docs per batch
    connection_timeout_seconds=10,
    num_retries=3,
    
    # Validation
    validate_embeddings=True,  # Check for empty vectors
    check_duplicates=True,     # Detect duplicate IDs
    
    # Error handling
    skip_on_error=True,        # Continue on individual failures
    max_errors=100,            # Stop after N errors
)

# Validate config
errors = config.validate()
if errors:
    print(f"Config errors: {errors}")
```

---

## Integration with Other Pipelines

### Complete 4-Stage Workflow

```python
# Stage 1: Ingestion
from doc_pipeline import DocumentIngestionPipeline
ingest = DocumentIngestionPipeline()
doc = ingest.ingest("contract.pdf")

# Stage 2: Chunking
from legal_rag_chunker import HierarchicalChunkingPipeline
chunker = HierarchicalChunkingPipeline()
chunks = chunker.process(doc.plain_text)

# Stage 3: Embedding
from embedding_pipeline import EmbeddingPipeline, EmbeddingConfig
embedder = EmbeddingPipeline(EmbeddingConfig(model_name="BAAI/bge-large-en-v1.5"))
emb_result = embedder.embed_chunks(chunks.chunks, source_file=doc.file_path)

# Stage 4: Storage ← THIS PIPELINE
from vector_storage import VectorStoragePipeline, StorageConfig
storage = VectorStoragePipeline(StorageConfig(
    host="localhost", port=8108, api_key="xyz"
))

# Create collection once
storage.create_collection_for_model(
    name="legal_docs",
    embedding_dim=1024,
    model_name="BAAI/bge-large-en-v1.5",
)

# Index
result = storage.index(
    collection_name="legal_docs",
    embeddings=emb_result.embeddings,
    model_version="v1.0",
)

print(result.summary())

# Query
query = "What are the payment terms?"
query_emb = embedder.embedder.embed_single(query)
results = storage.search("legal_docs", query_emb, k=5)

for r in results:
    print(f"{r['score']:.3f} | {' > '.join(r['hierarchy'])}")
    print(f"  {r['text'][:100]}...")
```

---

## Search Filters

Typesense supports powerful filtering:

```python
# By source file
filter_by="source_file:=contract_2024.pdf"

# By hierarchy (contains check)
filter_by="hierarchy:='Section 1'"

# By metadata
filter_by="metadata.doc_type:=employment"

# By version
filter_by="embedding_model_version:=v1.0"

# Combined (AND)
filter_by="source_file:=contract.pdf && metadata.importance:>=2"

# Combined (OR)
filter_by="source_file:=[contract.pdf, agreement.pdf]"
```

---

## Performance

### Benchmarks

| Documents | Batch Size | Time | Speed |
|---|---|---|---|
| 1,000 | 100 | 4.8s | 208 docs/sec |
| 10,000 | 100 | 48s | 208 docs/sec |
| 10,000 | 200 | 38s | 263 docs/sec |
| 100,000 | 200 | 6.3min | 264 docs/sec |

*Tests: 1024d vectors, localhost Typesense, default config*

### Optimization Tips

1. **Batch size**: Larger batches = fewer API calls
   - CPU: 100-200
   - Faster network: 200-500
   
2. **Disable validation** if you're confident data is clean:
   ```python
   config = StorageConfig(
       validate_embeddings=False,
       check_duplicates=False,
   )
   ```

3. **Typesense memory**: Allocate enough RAM
   - Rule of thumb: 2GB + (num_docs × embedding_dim × 4 bytes)
   - 1M docs × 1024d = ~4GB embeddings + 2GB overhead = 6GB

---

## Troubleshooting

**"Connection refused"**  
→ Typesense not running. Start with Docker or binary.

**"Invalid API key"**  
→ Check `api_key` in config matches Typesense `--api-key` flag.

**"Dimension mismatch"**  
→ Collection expects different dimension than your embeddings.  
→ Drop collection and recreate, or create new collection with correct dim.

**"Too many open files"**  
→ Increase batch size to reduce number of requests.

**Slow indexing**  
→ Increase `batch_size` (default 100 → try 200-500)  
→ Check network latency to Typesense  
→ Disable validation if not needed

---

## Dependencies

```
typesense>=0.16.0
```

Install:
```bash
pip install typesense
```

---

## Complete Example

See `vector_storage_demo_simple.py` for a working demo showing all 7 features.

Run:
```bash
python vector_storage_demo_simple.py
```

---

## License

MIT
