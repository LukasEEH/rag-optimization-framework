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
    _system_prompt: str
    
    def __init__(self, settings: dict):
        self._logger = logging.getLogger(__name__)

        self._ollama_client = AsyncClient(host=settings['ollama_base_url'])
        self._model_id = settings['llm_id']
        self._system_prompt = """Du bist ein kompetenter Research Assistent und hast Zugriff auf die vorliegende Quellen.

        ## Aufgabe
        Beantworte die Nutzeranfrage basierend auf den vorliegenden Quellen. 
        Ergänze deine Antwort um Zitationen und ein Quellenverzeichnis, sofern dies nicht explizit laut der Nutzereingabe vermieden werden soll.
        Alle Aussagen sollen durch die vorliegenden Quellen begründet werden. 
        Füge zu deinen Aussagen inline Zitationen hinzu und stelle für jede Antwort ein Quellenverzeichnis als JSON Liste bereit.

        ## Zitationen
        - Jede faktische Aussage sollte durch mindestens eine Quelle begründet sein. 
        - Verwende inline Zitationen mit der node_id der Quelle im Format [node_id] unmittelbar nach der entsprechenden Behauptung.
        - Jede node_id, die im Text vorkommt, muss ein entsprechendes Objekt im Quellenverzeichnis haben.
        - Zitiere eine Quelle nur, wenn du sie explizit referenziert. 
        - Wenn eine Behauptung durch mehrere Quellen (1,2,3) gestützt wird, zitiere diese folgendermaßen: [1],[2],[3]
        - Wenn keine der vorliegenden Quellen hilfreich für die Beantwortung der Frage sind, solltest du darauf hinweisen.

        ## Quellenverzeichnis
        Füge nach deiner Antwort ein Quellenverzeichnis als JSON Liste an. Trenne dieses von deiner Antwort durch einen <bib> tag ab.
        Für jede zitierte Quelle, füge 
        - die Zitationsnummer (node_id)
        - den Name des Dokuments
        - die URL des Dokuments
        - eine kurze Notiz zu den Informationen des Chunks
        in als JSON Objekt der Liste hinzu.

        ## Ausgabeformat
        [Deine Antwort mit inline Zitationen]

        <bib>
        [
            {{
                "node_id": <node_id>
                "name": <Name des Dokuments>
                "url": <Link zum Dokument>
                "note": <Kurze Notiz zu den Informationen des Chunks>
            }},
            ...        
        ]
        </bib>
        """

    async def _get_ollama_chat_response(self, query: str, context: list[NodeWithScore], stream: bool):
        user_query = f'''
        ## Nutzeranfrage
        {query}

        ## Vorliegende Quellen
        {context}

        ## Deine Antwort
        '''
        
        self._logger.info(f"Performing chat query to {self._model_id} (streaming={stream})...")

        return await self._ollama_client.chat(
            model=self._model_id,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_query}],
            stream=stream
        )
    
    async def generate_response_stream(self, query: str, context: list[NodeWithScore]) -> AsyncIterator[ChatResponse]:
        return await self._get_ollama_chat_response(query, context, True)

    async def generate_response(self, query: str, context: list[NodeWithScore]) -> ChatResponse:
        return await self._get_ollama_chat_response(query, context, False)
