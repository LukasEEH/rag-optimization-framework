import logging
from logging import Logger
from typing import List, Optional
from ollama import Client

from llama_index.core import QueryBundle, StorageContext, load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore 
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.retrievers import BaseRetriever as llama_index_BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor

from interfaces.base_retriever import BaseRetriever

class Retriever(BaseRetriever):
    _logger: Logger
    _dense_retriever: llama_index_BaseRetriever
    _ollama_client: Client
    _model_id: str
    _refinement_system_prompt: str

    def __init__(self, settings: dict):
        self._logger = logging.getLogger(__name__)

        index_dir = settings['index_dir']
        self._model_id = settings['llm_id']
        self._ollama_client = Client(host=settings['ollama_base_url'])

        self._dedupe_postprocessor = self.DeduplicationPostProcessor()
        self._rerank_postprocessor = SentenceTransformerRerank(
            model="nvidia/llama-nemotron-rerank-1b-v2",
            top_n=4
        )

        self._logger.info(f"Initializing storage context from index persistance directory {index_dir}")
        dense_storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(f"{index_dir}"),
            vector_store=SimpleVectorStore.from_persist_dir(f"{index_dir}"),
            index_store=SimpleIndexStore.from_persist_dir(f"{index_dir}"),
            graph_store=SimpleGraphStore.from_persist_dir(f"{index_dir}")
        )

        self._logger.info(f"Loading index from persistance directory {index_dir}")
        dense_index = load_index_from_storage(dense_storage_context)

        self._dense_retriever = dense_index.as_retriever(similarity_top_k=6)

        self._refinement_system_prompt = """Du bist ein Spezialist für die Identifikation von Informationslücken in den Kontext-Chunks eines RAG-Systems.

        **AUFGABE**
        Analysiere die gegebene Benutzeranfrage und den vorliegenden Kontext zu deren Beantwortung. 
        Identifiziere, welche Informationen zur vollständigen Beantwortung fehlen und formuliere, falls nötig, eine präzise Beschreibung der fehlenden Informationen, um diese gezielt abrufen zu können.
        Stelle sicher, dass du für komplexe Benutzeranfragen alle fehlenden Aspekte beschreibst und nutze den bereits vorhandenen Kontext zur Präzisierung.

        **AUSGABEFORMAT**
        - Falls weitere Informationen fehlen: Gib NUR eine prägnante Beschreibung der fehlenden Informationen in MAXIMAL 2 Sätzen aus 
        - Falls der Kontext zur Beantwortung der Frage ausreicht, gib NUR '<skip>' aus

        **BEISPIEL**
        Benutzeranfrage: "In welcher Stadt arbeitete der Mathematiker, der die Turing-Maschine konzipierte, während des Zweiten Weltkriegs?"
        Kontext: ["Die Turingmaschine ist benannt nach dem britischen Mathematiker Alan Turing, der sie 1936/37 einführte.", "Als Zweiter Weltkrieg (01.09.1939 bis 02.09.1945) wird der zweite global geführte Krieg sämtlicher Großmächte im 20. Jahrhundert bezeichnet."]
        Fehlende Informationen: "Wohnort von Alan Turing während des Zweiten Weltkriegs (September 1939 bis September 1945)."
        """


    def retrieve(self, query: str) -> list[NodeWithScore]:
        self._logger.info(f"Retrieving nodes from index...")
        dense_nodes = self._dense_retriever.retrieve(query)
        self._logger.info(f"Retrieved {len(dense_nodes)} nodes.")
        
        initial_chunks = [node.get_content() for node in dense_nodes]

        refinement_query = f"""Benutzeranfrage: "{query}"
        Kontext: {initial_chunks}
        Fehlende Informationen:"""

        self._logger.info("Check for query refinement...")
        response = self._ollama_client.chat(
            model=self._model_id,
            messages=[
                {"role": "system", "content": self._refinement_system_prompt},
                {"role": "user", "content": refinement_query}],
            stream=False
        )

        refined_query = response.message.content

        if "<skip>" in refined_query:
            self._logger.info("Skipping refined retrieval")
        else:
            self._logger.info("Refining retrieval...")
            additional_nodes = self._dense_retriever.retrieve(refined_query)
            dense_nodes += additional_nodes
            self._logger.info(f"Retrieved {len(additional_nodes)} additional nodes.")
        
        self._logger.info("Postprocess nodes...")
        deduped_nodes = self._dedupe_postprocessor.postprocess_nodes(dense_nodes)
        self._logger.info(f"Deduplicated nodes. Removed {len(dense_nodes) - len(deduped_nodes)} nodes, {len(deduped_nodes)} left")

        reranked_nodes = self._rerank_postprocessor.postprocess_nodes(deduped_nodes, query_str=query)
        self._logger.info(f"Reranked and pruned nodes. Removed {len(deduped_nodes) - len(reranked_nodes)} nodes, {len(reranked_nodes)} left")

        return reranked_nodes
    

    class DeduplicationPostProcessor(BaseNodePostprocessor):

        @classmethod
        def class_name(cls) -> str:
            return "DeduplicationPostProcessor"

        def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
        ) -> List[NodeWithScore]:
            seen_ids = set()
            filtered_nodes = []

            for n in nodes:
                node_id = n.node.node_id
                
                if node_id in seen_ids:
                    continue

                filtered_nodes.append(n)
                seen_ids.add(node_id)

            return filtered_nodes
