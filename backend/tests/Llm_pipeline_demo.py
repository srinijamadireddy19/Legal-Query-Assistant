"""
LLM Pipeline Demo
─────────────────
Shows all features:
✅ Query classification (new vs followup vs clarification)
✅ Optimized prompts for each scenario
✅ Context quality handling (high/medium/low/none)
✅ Threshold-based responses (< 0.3 = appropriate "not found" message)
✅ Conversation management
✅ Citation generation

NOTE: This demo uses mock components. In production, connect to real Groq API.
"""

import sys
sys.path.insert(0, "/home/claude")

from llm_pipeline.core.models import (
    LLMConfig,
    QueryContext,
    QueryType,
    ContextQuality,
    RetrievedContext,
    ConversationMessage,
)
from llm_pipeline.prompts.templates import PromptSelector
from llm_pipeline.core.classifier import QueryClassifier


print("╔" + "═" * 78 + "╗")
print("║" + " " * 25 + "LLM PIPELINE DEMO" + " " * 36 + "║")
print("╚" + "═" * 78 + "╝\n")

# ══════════════════════════════════════════════════════════════════════════
# Feature 1: Query Classification
# ══════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("✅ FEATURE 1: Query Type Classification")
print("=" * 80)

classifier = QueryClassifier()

# No history = NEW
query1 = "What are the payment terms in the contract?"
type1 = classifier.classify(query1, history=[])
print(f"\nQuery: '{query1}'")
print(f"Type: {type1.value} (no history)")

# Create history
history = [
    ConversationMessage(role="user", content=query1),
    ConversationMessage(role="assistant", content="The payment terms specify..."),
]

# Follow-up questions
test_queries = [
    ("What about late fees?", "FOLLOWUP - 'what about' keyword"),
    ("And what happens after 60 days?", "FOLLOWUP - starts with 'and'"),
    ("Can you clarify that?", "CLARIFICATION - explicit request"),
    ("What does net 30 mean?", "CLARIFICATION - asking definition"),
    ("Tell me more", "FOLLOWUP - continuation request"),
    ("Who is the landlord?", "NEW - completely different topic"),
]

print("\nWith conversation history:")
for query, explanation in test_queries:
    qtype = classifier.classify(query, history)
    print(f"  '{query}'")
    print(f"    → {qtype.value} ({explanation})")

# ══════════════════════════════════════════════════════════════════════════
# Feature 2: Context Quality Thresholds
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("✅ FEATURE 2: Context Quality Thresholds")
print("=" * 80)

contexts_examples = [
    (0.85, "HIGH", "Very relevant match"),
    (0.62, "MEDIUM", "Somewhat relevant"),
    (0.42, "LOW", "Marginally relevant"),
    (0.25, "NO_CONTEXT", "Below threshold (0.3)"),
]

print("\nContext quality based on similarity score:")
for score, quality, description in contexts_examples:
    ctx = RetrievedContext(
        text="Sample text",
        score=score,
        source_file="contract.pdf",
        hierarchy=["Section 1"],
        chunk_id="chunk_1",
    )
    print(f"  Score {score:.2f} → {ctx.quality.value.upper():11s} - {description}")

print(f"\nThreshold: {0.3} (configurable in LLMConfig)")
print("Below threshold → 'No relevant context' response")

# ══════════════════════════════════════════════════════════════════════════
# Feature 3: Optimized Prompts for Different Scenarios
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("✅ FEATURE 3: Optimized Prompts")
print("=" * 80)

# Create sample contexts
high_quality_ctx = RetrievedContext(
    text="Payment is due within 30 days of invoice date. Late fee of 5% applies after grace period.",
    score=0.89,
    source_file="contract.pdf",
    hierarchy=["Section 3", "Payment Terms"],
    chunk_id="chunk_12",
    page_numbers=[3],
)

low_quality_ctx = RetrievedContext(
    text="The parties agree to the terms outlined herein.",
    score=0.42,
    source_file="contract.pdf",
    hierarchy=["Section 1", "Introduction"],
    chunk_id="chunk_1",
)

# Scenario 1: NEW question with HIGH quality context
print("\n📋 Scenario 1: NEW question + HIGH quality context (score >= 0.7)")
print("─" * 80)

query_ctx_1 = QueryContext(
    query="What are the payment terms?",
    query_type=QueryType.NEW,
    contexts=[high_quality_ctx],
)

prompt_1 = PromptSelector.select_prompt(query_ctx_1)
print(f"Query: {query_ctx_1.query}")
print(f"Context quality: {query_ctx_1.context_quality.value}")
print(f"Prompt type: Standard Q&A with full context")
print(f"Prompt preview:\n{prompt_1[:200]}...\n")

# Scenario 2: NEW question with LOW quality context
print("\n📋 Scenario 2: NEW question + LOW quality context (0.3-0.5)")
print("─" * 80)

query_ctx_2 = QueryContext(
    query="What are the payment terms?",
    query_type=QueryType.NEW,
    contexts=[low_quality_ctx],
)

prompt_2 = PromptSelector.select_prompt(query_ctx_2)
print(f"Query: {query_ctx_2.query}")
print(f"Context quality: {query_ctx_2.context_quality.value}")
print(f"Prompt type: Uncertain context - explains limitations")
print(f"Prompt preview:\n{prompt_2[:250]}...\n")

# Scenario 3: NEW question with NO context (below threshold)
print("\n📋 Scenario 3: NEW question + NO context (score < 0.3)")
print("─" * 80)

query_ctx_3 = QueryContext(
    query="What is the employee's vacation policy?",
    query_type=QueryType.NEW,
    contexts=[],  # No contexts
)

prompt_3 = PromptSelector.select_prompt(query_ctx_3)
print(f"Query: {query_ctx_3.query}")
print(f"Context quality: {query_ctx_3.context_quality.value}")
print(f"Prompt type: No context found - polite explanation")
print(f"Prompt preview:\n{prompt_3[:300]}...\n")

# Scenario 4: FOLLOWUP question
print("\n📋 Scenario 4: FOLLOWUP question")
print("─" * 80)

history_with_previous = [
    ConversationMessage(role="user", content="What are the payment terms?"),
    ConversationMessage(role="assistant", content="Payment is due within 30 days of invoice date."),
]

query_ctx_4 = QueryContext(
    query="What about late fees?",
    query_type=QueryType.FOLLOWUP,
    contexts=[high_quality_ctx],
    history=history_with_previous,
)

prompt_4 = PromptSelector.select_prompt(query_ctx_4)
print(f"Query: {query_ctx_4.query}")
print(f"Context quality: {query_ctx_4.context_quality.value}")
print(f"Prompt type: Follow-up with previous context")
print(f"Prompt preview:\n{prompt_4[:250]}...\n")

# ══════════════════════════════════════════════════════════════════════════
# Feature 4: Appropriate "Not Found" Responses
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("✅ FEATURE 4: Appropriate 'Not Found' Responses")
print("=" * 80)

print("\nWhen similarity score < 0.3:")
print("  ❌ BAD: 'I don't know'")
print("  ❌ BAD: Making up information")
print("  ✅ GOOD: Helpful explanation:")

example_not_found = """
I was unable to find information about [topic] in the available documents.

The current document collection includes:
- Employment contracts
- Service agreements
- Non-disclosure agreements

To help you better, you could:
1. Rephrase your question to be more specific
2. Check if the relevant document has been indexed
3. Specify which document you're referring to

Would you like to try rephrasing your question?
"""

print(example_not_found)

# ══════════════════════════════════════════════════════════════════════════
# Feature 5: Citation Generation
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("✅ FEATURE 5: Citation Generation")
print("=" * 80)

contexts_with_citations = [
    RetrievedContext(
        text="Payment is due within 30 days...",
        score=0.89,
        source_file="service_agreement.pdf",
        hierarchy=["Section 3", "Payment Terms"],
        chunk_id="chunk_12",
        page_numbers=[3, 4],
    ),
    RetrievedContext(
        text="Late fee of 5% applies...",
        score=0.82,
        source_file="service_agreement.pdf",
        hierarchy=["Section 3", "Payment Terms", "3.2 Late Fees"],
        chunk_id="chunk_13",
        page_numbers=[4],
    ),
]

query_ctx_cit = QueryContext(
    query="What are the payment terms?",
    query_type=QueryType.NEW,
    contexts=contexts_with_citations,
)

citations = query_ctx_cit.get_citations(max_citations=3)

print("\nGenerated citations:")
for i, citation in enumerate(citations, 1):
    print(f"  [{i}] {citation}")

print("\nCitations are appended to the response:")
print("─" * 80)
print("Payment is due within 30 days of invoice date...")
print("\n**Sources:**")
for i, citation in enumerate(citations, 1):
    print(f"{i}. {citation}")

# ══════════════════════════════════════════════════════════════════════════
# Complete API Example
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("COMPLETE API USAGE")
print("=" * 80)

print("""
from llm_pipeline import LLMPipeline, LLMConfig, GroqModels
from embedding_pipeline import EmbeddingPipeline
from vector_storage import VectorStoragePipeline

# Initialize components
embedder = EmbeddingPipeline(...)
storage = VectorStoragePipeline(...)

# Create LLM pipeline
config = GroqModels.llama_70b_versatile(api_key="your-key")
config.context_score_threshold = 0.3  # Minimum score
config.max_context_length = 5         # Top 5 contexts

llm = LLMPipeline(config, embedder, storage)

# First question (NEW)
response = llm.query(
    query="What are the payment terms?",
    collection_name="legal_docs",
)

print(response.answer)
print(f"Quality: {response.context_quality.value}")
print(f"Citations: {response.citations}")

# Follow-up question (automatically detected)
response2 = llm.query(
    query="What about late fees?",
    collection_name="legal_docs",
)

print(response2.answer)
print(f"Query type: {response2.query_type.value}")  # FOLLOWUP

# Question with no context found
response3 = llm.query(
    query="What is the vacation policy?",
    collection_name="legal_docs",
)

# Will get appropriate "not found" message
print(response3.answer)
print(f"Quality: {response3.context_quality.value}")  # NO_CONTEXT
""")

print("\n" + "=" * 80)
print("✅ All Features Demonstrated!")
print("=" * 80)

print("\nKey capabilities:")
print("  ✓ Automatic query classification (new/followup/clarification)")
print("  ✓ Context quality thresholds (high/medium/low/none)")
print("  ✓ Optimized prompts per scenario")
print("  ✓ Appropriate 'not found' messages (score < 0.3)")
print("  ✓ Citation generation")
print("  ✓ Conversation history management")