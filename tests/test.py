from llm_pipeline.pipeline import LLMPipeline
from llm_pipeline.providers.groq_provider import GroqModels
from llm_pipeline.embedder import Embedder
from llm_pipeline.storage import Storage
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm_pipeline = LLMPipeline(
    GroqModels.llama_70b_versatile(GROQ_API_KEY),
    embedder,
    storage
)

r1 = llm.query("What are the payment terms?", "legal_docs")
print(r1.answer)
print(f"Type: {r1.query_type.value}")  # NEW
print(f"Quality: {r1.context_quality.value}")  # HIGH/MEDIUM/LOW/NO_CONTEXT
print(f"Citations: {r1.citations}")

# Follow-up (automatically detected!)
r2 = llm.query("What about late fees?", "legal_docs")
print(r2.answer)
print(f"Type: {r2.query_type.value}")  # FOLLOWUP

# No context found (score < 0.3)
r3 = llm.query("What is the vacation policy?", "legal_docs")
print(r3.answer)
