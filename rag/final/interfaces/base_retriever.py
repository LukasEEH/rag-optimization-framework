from abc import ABC, abstractmethod
from llama_index.core.schema import NodeWithScore

class BaseRetriever(ABC):
    """The retriever component for a RAG system."""

    @abstractmethod
    def retrieve(self, query: str) -> list[NodeWithScore]:
        """Retrieving nodes based on the query."""
        pass
