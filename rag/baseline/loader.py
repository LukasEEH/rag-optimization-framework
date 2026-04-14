import logging
from logging import Logger

from interfaces.base_loader import BaseLoader

from llama_index.core.readers import SimpleDirectoryReader
from llama_index.readers.confluence import ConfluenceReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import Document


class Loader(BaseLoader):
    _logger: Logger
    _sources: dict

    def __init__(self, sources_dict: dict):
        self._logger = logging.getLogger(__name__)
        self._sources = sources_dict
        

    def get_documents(self) -> list[Document]:
        self._logger.info("Loading documents...")
        docs = []
        for source in self._sources:
            if source['type'] == 'files':
                docs += self.handle_local_files(source['dirs'])
                self._logger.info(f"Added file docs, now {len(docs)} docs added.")
            elif source['type'] == 'websites':
                docs += self.handle_websites(source['urls'])
                self._logger.info(f"Added website docs, now {len(docs)} docs added.")
            elif source['type'] == 'confluence_sites':
                if 'space_key' in source:
                    docs += self.handle_confluence_space(source['base_url'], source['api_token'], source['space_key'])
                else:
                    docs += self.handle_confluence_sites(source['base_url'], source['api_token'], source['page_ids'])
                self._logger.info(f"Added confluence docs, now {len(docs)} docs added.")
            else:
                self._logger.warning(f"No handle defined for source from type {source['type']}.")
        return docs


    def handle_local_files(self, dir_paths: list[str]) -> list[Document]:
        docs = []
        for path in dir_paths:
            docs += SimpleDirectoryReader(path).load_data()
        return docs


    def handle_websites(self, urls: list[str]) -> list[Document]:
        docs = []
        docs += SimpleWebPageReader(html_to_text=True).load_data(urls=urls)
        return docs


    def handle_confluence_sites(self, base_url: str, api_token: str, page_ids: list[str]) -> list[Document]:
        docs = []
        reader = ConfluenceReader(
            base_url=base_url,
            api_token=api_token,
            client_args={'verify_ssl': False}
        )

        docs += reader.load_data(page_ids=page_ids, include_attachments=True)
        return docs
    

    def handle_confluence_space(self, base_url: str, api_token: str, space_key: str) -> list[Document]:
        docs = []
        reader = ConfluenceReader(
            base_url=base_url,
            api_token=api_token,
            client_args={'verify_ssl': False}
        )

        docs += reader.load_data(space_key=space_key, include_attachments=True, limit = 1000)
        return docs
    