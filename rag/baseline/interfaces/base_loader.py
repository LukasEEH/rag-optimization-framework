from abc import ABC, abstractmethod
from llama_index.core import Document

class BaseLoader(ABC):
    """The loader component for a RAG system"""

    @abstractmethod
    def get_documents(self) -> list[Document]:
        """Loads documents from one or multiple datasources."""
        pass
