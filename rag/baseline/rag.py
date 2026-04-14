import logging
from logging import Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

import json
from typing import AsyncIterator
from ollama import ChatResponse

from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

from interfaces.base_loader import BaseLoader
from interfaces.base_indexer import BaseIndexer
from interfaces.base_retriever import BaseRetriever
from interfaces.base_generator import BaseGenerator

from loader import Loader
from indexer import Indexer
from retriever import Retriever
from generator import Generator


class RAG():
    _logger: Logger
    _config_dict: dict

    _loader: BaseLoader
    _indexer: BaseIndexer
    _retriever: BaseRetriever
    _generator: BaseGenerator

    def __init__(self, config_path: str):
        self._logger = logging.getLogger(__name__)

        self._logger.info(f"Loading RAG configuration from {config_path}...")
        with open(config_path, 'r') as config_file:
            self._config_dict = json.load(config_file)

        self._logger.info("Reading model configuration from config...")
        ollama_base_url = self._config_dict['settings']['ollama_base_url']
        embedding_model_id = self._config_dict['settings']['embedding_model_id']
        llm_id = self._config_dict['settings']['llm_id']

        self._logger.info("Setting default models...")
        ollama_embedding = OllamaEmbedding(
            model_name=embedding_model_id,
            base_url=ollama_base_url
        )

        ollama_llm = Ollama(
            model=llm_id,
            base_url=ollama_base_url,
            streaming=True,
            thinking=True,
            request_timeout=600.0
        )

        Settings.embed_model = ollama_embedding
        self._logger.info(f"Set {ollama_embedding.model_name} as default embedding model.")

        Settings.llm = ollama_llm
        self._logger.info(f"Set {ollama_llm.model} as default llm.")

        self._logger.info("Reading index persistance directory from config...")
        index_dir = self._config_dict['settings']['index_dir']
        
        self._indexer = Indexer(index_dir)

        self._logger.info(f"Checking whether the index could be found in {index_dir}...")
        
        if not self._indexer.index_exists():
            self._logger.info(f"No index could be found in {index_dir}. Starting index creation...")
            self._loader = Loader(self._config_dict['sources'])
            docs = self._loader.get_documents()
            self._indexer.create_index(docs)

        self._retriever = Retriever(self._config_dict['settings'])
        self._generator = Generator(self._config_dict['settings'])


    async def respond(self, query: str) -> ChatResponse:
        context = self._retriever.retrieve(query)
        return await self._generator.generate_response(query, context)


    async def respond_with_stream(self, query: str) -> AsyncIterator[ChatResponse]:
        context = self._retriever.retrieve(query)
        return await self._generator.generate_response_stream(query, context)


    async def respond_verbose(self, query: str) -> tuple[str, list[NodeWithScore]]:
        context = self._retriever.retrieve(query)
        response = await self._generator.generate_response(query, context)
        return response, context
