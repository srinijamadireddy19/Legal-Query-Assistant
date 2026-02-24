"""
Base embedder interface and registry.
All embedding models implement BaseEmbedder.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Type, Optional
import logging

from .models import ChunkEmbedding, EmbeddingConfig, ModelInfo

log = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding models.
    Provides common interface for FastEmbed, SentenceTransformers, OpenAI, etc.
    """
    
    # Subclasses set this to their supported model names
    SUPPORTED_MODELS: List[str] = []
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self.model_info: Optional[ModelInfo] = None
    
    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the embedding model into memory.
        Returns True on success, False on failure.
        Must set self.model and self.model_info.
        """
        pass
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        Returns list of embedding vectors (same length as input).
        """
        pass
    
    @abstractmethod
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """Get information about the loaded model."""
        return self.model_info
    
    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded() else "not loaded"
        return f"{self.__class__.__name__}({self.config.model_name}, {status})"


class EmbedderRegistry:
    """
    Registry mapping model names to embedder classes.
    Supports pattern matching (e.g., "BAAI/*" → FastEmbedder).
    """
    
    def __init__(self):
        self._registry: Dict[str, Type[BaseEmbedder]] = {}
        self._patterns: List[tuple[str, Type[BaseEmbedder]]] = []
    
    def register(self, embedder_class: Type[BaseEmbedder] = None, pattern: Optional[str] = None):
        """
        Register an embedder class.

        Supports two call styles:
          - Direct:    registry.register(MyEmbedder, pattern="BAAI/*")
          - Decorator: @registry.register(pattern="BAAI/*")

        Args:
            embedder_class: The embedder class (omit when used as @register(pattern=...))
            pattern: Optional glob pattern for matching model names (e.g., "BAAI/*")
        """
        def _do_register(cls: Type[BaseEmbedder]) -> Type[BaseEmbedder]:
            # Register explicit model names
            for model_name in cls.SUPPORTED_MODELS:
                self._registry[model_name.lower()] = cls
            # Register pattern if provided
            if pattern:
                self._patterns.append((pattern.lower(), cls))
            return cls

        if embedder_class is not None:
            # Called as registry.register(MyEmbedder) or registry.register(MyEmbedder, pattern=...)
            return _do_register(embedder_class)
        else:
            # Called as @registry.register(pattern="BAAI/*") — return a decorator
            return _do_register
    
    def get(self, model_name: str) -> Type[BaseEmbedder]:
        """
        Get embedder class for a model name.
        Tries exact match first, then pattern matching.
        """
        key = model_name.lower()
        
        # Exact match
        if key in self._registry:
            return self._registry[key]
        
        # Pattern matching
        for pattern, embedder_cls in self._patterns:
            if self._matches_pattern(key, pattern):
                return embedder_cls
        
        raise ValueError(
            f"No embedder registered for model '{model_name}'. "
            f"Registered: {list(self._registry.keys())[:5]}..."
        )
    
    def _matches_pattern(self, model_name: str, pattern: str) -> bool:
        """Check if model_name matches pattern (e.g., "baai/*")."""
        if pattern.endswith("/*"):
            prefix = pattern[:-2]
            return model_name.startswith(prefix)
        return model_name == pattern
    
    def list_supported_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._registry.keys())
    
    def list_embedders(self) -> List[str]:
        """List all registered embedder classes."""
        return list(set(cls.__name__ for cls in self._registry.values()))


# Global registry instance
registry = EmbedderRegistry()