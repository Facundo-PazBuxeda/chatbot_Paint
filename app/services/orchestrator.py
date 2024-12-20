from typing import Dict, Optional
from langchain_cohere import ChatCohere
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import AsyncIteratorCallbackHandler
from models.models import IntentType, MessageIntent, IntentClassificationError
from core.config import settings

import json
import logging

logger = logging.getLogger(__name__)

class OrchestratorService:
    def __init__(self, cohere_api_key: str, rag_service, sql_service):
        self.llm = ChatCohere(cohere_api_key=cohere_api_key, model="command-r-plus-08-2024", seed= 42, temperature=0.3)
        self.rag_service = rag_service
        self.sql_service = sql_service
        self.recent_messages = []
        self._setup_prompts()

    def _setup_prompts(self):
        self.router_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Clasifica el mensaje del cliente de pinturería:
            1. CHAT: Saludos, despedidas o charla informal
            2. RAG: Preguntas sobre teoría de pinturas, técnicas, consejos, herramientas
            3. SQL: Preguntas sobre productos específicos, precios, stock, marcas o categorías
            
            Ejemplos SQL:
            - "¿Cuánto cuesta la pintura Alba?"
            - "¿Tienen pinturas látex?"
            - "Busco pinturas Sherwin Williams"
            - "¿Qué marcas de esmalte tienen?"
            - "¿Hay ofertas en pinturas?"
            
            Ejemplos RAG:
            - "¿Que tipos de pintura hay?"
            - "¿Que herramientas se usan para pintar?"
            - "Cuál es la diferencia entre pintura interior y pintura exterior?"
            - "Mejores metodos de pintura"
            
            Response JSON FORMAT:
            
            {
                "type": "CHAT|RAG|SQL",
                "confidence": float,
                "metadata": {"keywords": [], "entities": []}
            }"""),
            HumanMessage(content="{input}")
        ])

    async def process_message(self, message: str, user_id: Optional[str] = None) -> str:
        try:
            self.recent_messages = (self.recent_messages + [message])[-10:]
            intent = await self._classify_intent(message)
            logger.info(f"Classified intent: {intent.dict()}")

            if intent.type == IntentType.CHAT:
                response = await self._handle_chat(message)
            elif intent.type == IntentType.RAG:
                response = await self.rag_service.get_response(message)
            elif intent.type == IntentType.SQL:
                response = await self.sql_service.get_response(message)
            else:
                response = "No pude entender tu consulta. ¿Podrías reformularla?"

            return response

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return "Ocurrió un error. Por favor, intenta nuevamente."

    async def _classify_intent(self, message: str) -> MessageIntent:
        chain = self.router_prompt | self.llm
        response = await chain.ainvoke({"input": message})
        intent_data = json.loads(response.content)
        return MessageIntent(**intent_data)

    async def _handle_chat(self, message: str) -> str:
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Eres un asistente amable de una pinturería.
                          DEBES DECIDIR ENTRE 3 OPCIONES:
                          1. RAG: Preguntas sobre teoría de pinturas, técnicas, consejos, herramientas
                          2. SQL: Preguntas sobre productos específicos, precios, stock, marcas o categorías
                          3. CHAT: Saludos, despedidas o charla informal"""),
            HumanMessage(content=message)
        ])
        chain = chat_prompt | self.llm
        response = await chain.ainvoke({})
        return response.content