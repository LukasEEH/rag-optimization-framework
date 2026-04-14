import logging
from logging import Logger
from typing import AsyncIterator
from ollama import AsyncClient, ChatResponse
from llama_index.core.schema import NodeWithScore

from interfaces.base_generator import BaseGenerator


class Generator(BaseGenerator):
    _logger: Logger
    _ollama_client: AsyncClient
    _model_id: str
    
    def __init__(self, settings: dict):
        self._logger = logging.getLogger(__name__)
        self._ollama_client = AsyncClient(host=settings['ollama_base_url'])
        self._model_id = settings['llm_id']

    async def _get_ollama_chat_response(self, query: str, context: list[NodeWithScore], stream: bool):
        user_query = f"Answer the user query \"{query}\" based on the following context: {context}."
        
        self._logger.info(f"Performing chat query to {self._model_id} (streaming={stream})...")

        return await self._ollama_client.chat(
            model=self._model_id,
            messages=[
                {"role": "user", "content": user_query}],
            stream=stream
        )
    
    async def generate_response_stream(self, query: str, context: list[NodeWithScore]) -> AsyncIterator[ChatResponse]:
        return await self._get_ollama_chat_response(query, context, True)

    async def generate_response(self, query: str, context: list[NodeWithScore]) -> ChatResponse:
        return await self._get_ollama_chat_response(query, context, False)
