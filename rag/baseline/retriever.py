import logging
from logging import Logger

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore 
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.retrievers import BaseRetriever as llama_index_BaseRetriever
from llama_index.core.schema import NodeWithScore

from interfaces.base_retriever import BaseRetriever

class Retriever(BaseRetriever):
    _logger: Logger
    _dense_retriever: llama_index_BaseRetriever
    _model_id: str

    def __init__(self, settings: dict):
        self._logger = logging.getLogger(__name__)

        index_dir = settings['index_dir']
        self._model_id = settings['llm_id']

        self._logger.info(f"Initializing storage context from index persistance directory {index_dir}")
        dense_storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(f"{index_dir}"),
            vector_store=SimpleVectorStore.from_persist_dir(f"{index_dir}"),
            index_store=SimpleIndexStore.from_persist_dir(f"{index_dir}"),
            graph_store=SimpleGraphStore.from_persist_dir(f"{index_dir}")
        )

        self._logger.info(f"Loading index from persistance directory {index_dir}")
        dense_index = load_index_from_storage(dense_storage_context)

        self._dense_retriever = dense_index.as_retriever(similarity_top_k=4)


    def retrieve(self, query: str) -> list[NodeWithScore]:
        self._logger.info(f"Retrieving nodes from index...")
        dense_nodes = self._dense_retriever.retrieve(query)
        self._logger.info(f"Retrieved {len(dense_nodes)} nodes.")
        return dense_nodes