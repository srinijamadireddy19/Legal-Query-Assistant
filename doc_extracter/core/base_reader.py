"""
Base reader interface and reader registry.
Every format-specific reader inherits BaseReader.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Type, List

from .models import DocumentResult


class BaseReader(ABC):
    """All readers must implement this interface."""

    #: file extensions this reader handles, lowercase without dot
    SUPPORTED_EXTENSIONS: List[str] = []

    @abstractmethod
    def read(self, file_path: str, **kwargs) -> DocumentResult:
        """
        Parse file_path and return a DocumentResult.
        Must never raise — capture exceptions into result.errors and
        return a FAILED/PARTIAL status.
        """


# ── Registry ─────────────────────────────────────────────────────────────────

class ReaderRegistry:
    """Maps file extensions → reader classes."""

    def __init__(self):
        self._registry: Dict[str, Type[BaseReader]] = {}

    def register(self, reader_class: Type[BaseReader]):
        for ext in reader_class.SUPPORTED_EXTENSIONS:
            self._registry[ext.lower()] = reader_class
        return reader_class                 # allow use as decorator

    def get(self, extension: str) -> Type[BaseReader]:
        ext = extension.lower().lstrip(".")
        reader_cls = self._registry.get(ext)
        if reader_cls is None:
            raise ValueError(
                f"No reader registered for '.{ext}'. "
                f"Supported: {list(self._registry.keys())}"
            )
        return reader_cls

    @property
    def supported_extensions(self) -> List[str]:
        return list(self._registry.keys())


# Singleton — importable everywhere
registry = ReaderRegistry()
