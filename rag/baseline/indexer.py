import logging
from logging import Logger
from pathlib import Path
from os.path import exists, join

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.node_parser.interface import TextSplitter

class Indexer:
    _logger: Logger
    _persist_dir: str
    _splitter: TextSplitter

    def __init__(self, persist_dir: str):
        self._logger = logging.getLogger(__name__)
        self._persist_dir = persist_dir
        self._splitter = SentenceSplitter.from_defaults(chunk_size=512, chunk_overlap=100)

    def index_exists(self) -> bool:
        required_files = ["docstore.json", "index_store.json", "default__vector_store.json"]
        index_exists = all(exists(join(f"{self._persist_dir}", f)) for f in required_files)
        return index_exists

    def create_index(self, docs: list[Document]):
        self._logger.info("Start splitting documents...")
        sentence_nodes = self._splitter.get_nodes_from_documents(docs)
        self._logger.info(f"Finished chunking. Created {len(sentence_nodes)} nodes from {len(docs)} documents.")

        self._logger.info("Start creating dense index...")
        doc_index = VectorStoreIndex(sentence_nodes, show_progress=True)
        self._logger.info("Finished vectorization and index creation.")

        self._logger.info("Preparing index persistance...")
        persist_path = Path(self._persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        self._logger.info("Persisting index...")
        doc_index.storage_context.persist(f"{self._persist_dir}")
        self._logger.info(f"Persisted index to {self._persist_dir}.")
