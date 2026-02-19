"""
FastEmbed Embedder
──────────────────
Uses the fastembed library (ONNX-based, fast inference).
Supports BAAI/bge-* models, intfloat/e5-*, sentence-transformers/*.

Advantages:
  - Fast inference (ONNX runtime)
  - Low memory footprint
  - No PyTorch dependency
  - Quantized models available

Installation:
  pip install fastembed
"""

import logging
from typing import List, Optional
import time

from ..core.base_embedder import BaseEmbedder, registry
from ..core.models import EmbeddingConfig, ModelInfo

log = logging.getLogger(__name__)


@registry.register(pattern="BAAI/*")
@registry.register(pattern="intfloat/*")
@registry.register(pattern="sentence-transformers/*")
class FastEmbedder(BaseEmbedder):
    """
    Embedder using fastembed library.
    Covers BAAI/bge-*, intfloat/e5-*, sentence-transformers/* models.
    """
    
    SUPPORTED_MODELS = [
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-base-en-v1.5", 
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-m3",
        "intfloat/e5-large-v2",
        "intfloat/e5-base-v2",
        "intfloat/e5-small-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ]
    
    # Model dimension mapping
    MODEL_DIMS = {
        "BAAI/bge-large-en-v1.5": 1024,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-m3": 1024,
        "intfloat/e5-large-v2": 1024,
        "intfloat/e5-base-v2": 768,
        "intfloat/e5-small-v2": 384,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
    }
    
    def load_model(self) -> bool:
        """Load FastEmbed model."""
        try:
            from fastembed import TextEmbedding
            
            log.info(f"Loading FastEmbed model: {self.config.model_name}")
            
            # Map model name to FastEmbed name (some differences)
            fastembed_name = self._map_model_name(self.config.model_name)
            
            self.model = TextEmbedding(
                model_name=fastembed_name,
                cache_dir=self.config.cache_dir,
                max_length=self.config.max_length,
            )
            
            # Set model info
            dimension = self.MODEL_DIMS.get(self.config.model_name, 768)
            self.model_info = ModelInfo(
                name=self.config.model_name,
                dimension=dimension,
                max_seq_length=self.config.max_length,
                description=f"FastEmbed ONNX model: {fastembed_name}",
            )
            
            log.info(f"✓ Loaded {self.model_info}")
            return True
            
        except ImportError:
            log.error("fastembed not installed. Install: pip install fastembed")
            return False
        except Exception as e:
            log.error(f"Failed to load FastEmbed model: {e}")
            return False
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not texts:
            return []
        
        try:
            # FastEmbed returns generator of numpy arrays
            embeddings = list(self.model.embed(
                texts,
                batch_size=self.config.batch_size,
            ))
            
            # Convert numpy to list of floats
            result = [embedding.tolist() for embedding in embeddings]
            
            # L2 normalize if configured
            if self.config.normalize:
                result = self._normalize_embeddings(result)
            
            return result
            
        except Exception as e:
            log.error(f"Embedding generation failed: {e}")
            # Return empty embeddings for failed texts
            return [[] for _ in texts]
    
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        result = self.embed([text])
        return result[0] if result else []
    
    def _map_model_name(self, model_name: str) -> str:
        """
        Map our model name to FastEmbed's naming convention.
        FastEmbed uses different names for some models.
        """
        # FastEmbed naming conventions
        mapping = {
            "BAAI/bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
            "BAAI/bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
            "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
            "BAAI/bge-m3": "BAAI/bge-m3",
            "intfloat/e5-large-v2": "intfloat/e5-large-v2",
            "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        }
        return mapping.get(model_name, model_name)
    
    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """L2 normalize embeddings for cosine similarity."""
        import math
        
        normalized = []
        for emb in embeddings:
            norm = math.sqrt(sum(x * x for x in emb))
            if norm > 0:
                normalized.append([x / norm for x in emb])
            else:
                normalized.append(emb)
        
        return normalized


# Add instruction/query prefixes for models that need them
class InstructionAwareEmbedder(FastEmbedder):
    """
    Extension of FastEmbedder that adds instruction prefixes.
    Some models (E5, BGE-M3) perform better with task-specific prefixes.
    """
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.query_prefix = ""
        self.passage_prefix = ""
        
        # Set prefixes based on model
        if "e5-" in config.model_name.lower():
            self.query_prefix = "query: "
            self.passage_prefix = "passage: "
        elif "bge-m3" in config.model_name.lower():
            # BGE-M3 uses different instructions
            self.query_prefix = "Represent this sentence for searching relevant passages: "
            self.passage_prefix = ""
    
    def embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Embed with instruction prefix.
        
        Args:
            texts: List of texts to embed
            is_query: If True, use query prefix. If False, use passage prefix.
        """
        prefix = self.query_prefix if is_query else self.passage_prefix
        
        if prefix:
            prefixed_texts = [prefix + text for text in texts]
        else:
            prefixed_texts = texts
        
        return super().embed(prefixed_texts)
    
    def embed_single(self, text: str, is_query: bool = False) -> List[float]:
        """Embed single text with instruction prefix."""
        result = self.embed([text], is_query=is_query)
        return result[0] if result else []