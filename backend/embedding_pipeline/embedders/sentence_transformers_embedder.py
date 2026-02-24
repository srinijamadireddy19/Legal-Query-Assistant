"""
SentenceTransformers Embedder
──────────────────────────────
Fallback embedder using sentence-transformers library.
Used when FastEmbed fails or for models not supported by FastEmbed.

Installation:
  pip install sentence-transformers
"""

import logging
from typing import List
import time

from ..core.base_embedder import BaseEmbedder, registry
from ..core.models import EmbeddingConfig, ModelInfo

log = logging.getLogger(__name__)


@registry.register
class SentenceTransformersEmbedder(BaseEmbedder):
    """
    Fallback embedder using sentence-transformers.
    Covers all HuggingFace sentence-transformer models.
    """
    
    SUPPORTED_MODELS = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "intfloat/e5-large-v2",
        "intfloat/e5-base-v2",
    ]
    
    def load_model(self) -> bool:
        """Load SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            log.info(f"Loading SentenceTransformer: {self.config.model_name}")
            
            self.model = SentenceTransformer(
                self.config.model_name,
                cache_folder=self.config.cache_dir,
                device=self.config.device,
            )
            
            # Get model info
            dimension = self.model.get_sentence_embedding_dimension()
            max_length = self.model.max_seq_length
            
            self.model_info = ModelInfo(
                name=self.config.model_name,
                dimension=dimension,
                max_seq_length=max_length,
                description=f"SentenceTransformer model from HuggingFace",
            )
            
            log.info(f"✓ Loaded {self.model_info}")
            return True
            
        except ImportError:
            log.error(
                "sentence-transformers not installed. "
                "Install: pip install sentence-transformers"
            )
            return False
        except Exception as e:
            log.error(f"Failed to load SentenceTransformer: {e}")
            return False
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not texts:
            return []
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=self.config.show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize,
            )
            
            # Convert to list of lists
            return embeddings.tolist()
            
        except Exception as e:
            log.error(f"Embedding generation failed: {e}")
            return [[] for _ in texts]
    
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            embedding = self.model.encode(
                text,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize,
            )
            return embedding.tolist()
        except Exception as e:
            log.error(f"Embedding generation failed: {e}")
            return []