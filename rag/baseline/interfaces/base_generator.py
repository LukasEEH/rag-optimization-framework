from abc import ABC, abstractmethod
from typing import AsyncIterator
from llama_index.core.schema import NodeWithScore

class BaseGenerator(ABC):
    "The generator component for a RAG system."
    
    @abstractmethod
    def generate_response(self, query: str, context: list[NodeWithScore]) -> any:
        """Generating a response for the query. Using the context for augmentation."""
        pass
    
    @abstractmethod
    def generate_response_stream(self, query: str, context: list[NodeWithScore]) -> AsyncIterator[any]:
        """Generating a response stream from the query. Using the context for augmentation."""
        pass
