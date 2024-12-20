from langchain_cohere import ChatCohere
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from core.config import settings
from typing import List, Optional
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from models.models import RAGProcessingError
import logging

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(
        self, 
        chroma_client: Chroma, 
        cohere_api_key: str,
        k_documents: int = 3,
        relevance_threshold: float = 0.7
    ):
        self.vectorstore = chroma_client
        self.llm = ChatCohere(cohere_api_key=cohere_api_key, model="command-r-plus-08-2024", temperature=0, seed=42)
        self.k_documents = k_documents
        self.relevance_threshold = relevance_threshold
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.k_documents}
        )
        self._setup_prompts()
        
    def _setup_prompts(self):
        self.context_relevance_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Evalúa la relevancia del contexto para la pregunta.
            
            Pregunta: {question}
            Contexto: {context}
            
            Responde con un número entre 0 y 1."""),
            HumanMessage(content="¿Qué tan relevante es este contexto?")
        ])
        
        self.response_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Eres un experto en pinturas. Genera una respuesta
            precisa usando el contexto y conocimiento previo.
            
            Contexto del historial: {history_context}
            
            Documentación relevante:
            {relevant_docs}"""),
            HumanMessage(content="{question}")
        ])

    async def get_response(
        self, 
        question: str,
        history_context: Optional[str] = ""
    ) -> str:
        try:
            docs = await self._get_relevant_documents(question)
            
            if not docs:
                return "Lo siento, no encontré información específica sobre eso. ¿Podrías reformular tu pregunta?"
            
            chain = self.response_prompt | self.llm
            response = await chain.ainvoke({
                "question": question,
                "history_context": history_context,
                "relevant_docs": "\n\n".join(docs)
            })
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error in RAG processing: {str(e)}")
            raise RAGProcessingError(
                "Error procesando la consulta",
                details={"error": str(e)}
            )

    async def _get_relevant_documents(self, question: str) -> List[str]:
        docs = await self.retriever.aget_relevant_documents(question)
        
        relevant_docs = []
        for doc in docs:
            chain = self.context_relevance_prompt | self.llm
            relevance = await chain.ainvoke({
                "question": question,
                "context": doc.page_content
            })
            
            try:
                relevance_score = float(relevance.content)
                if relevance_score >= self.relevance_threshold:
                    relevant_docs.append(doc.page_content)
            except ValueError:
                logger.warning(f"Invalid relevance score: {relevance.content}")
                
        return relevant_docs