from abc import ABC, abstractmethod
from llama_index.core import Document

class BaseIndexer(ABC):
    """The indexer component for a RAG system"""

    @abstractmethod
    def index_exists(self, dir: str) -> bool:
        """Checks if an index exists in the specified dir."""
        pass

    @abstractmethod
    def create_index(self, docs: list[Document]) -> None:
        """Creates an index from the given documents."""
        pass
