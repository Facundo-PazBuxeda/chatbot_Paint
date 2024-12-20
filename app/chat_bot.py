from typing import Annotated, Dict, List, TypedDict
from langchain_chroma import Chroma
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
import operator
import logging
from core.config import settings

logger = logging.getLogger(__name__)


class ChatState(TypedDict):
    messages: List[Dict[str, str]]
    next_step: str
    current_user: str
    response: str
    llm: ChatCohere


def detect_intent(state: ChatState) -> Dict:
    last_message = state["messages"][-1]["content"]
    prompt = """Clasifica si el siguiente mensaje es small talk o una consulta.
    Small talk incluye: saludos, despedidas, agradecimientos, charla casual.
    Consulta incluye: preguntas sobre productos o guías, búsqueda de información.
    
    Mensaje: {last_message}
    
    Responde SOLO con 'small_talk' o 'consulta'.
    """.format(
        last_message=last_message
    )

    result = state["llm"].predict(prompt).lower()
    logger.error(f"Select collection result: {result}")
    return {
        "next_step": (
            "handle_small_talk" if "small_talk" in result else "select_collection"
        )
    }


def select_collection(state: ChatState) -> Dict:
    last_message = state["messages"][-1]["content"]
    prompt = """Clasifica si el siguiente mensaje requiere información de guías o productos.
    
    Guías incluye:
    - Tutoriales o instrucciones
    - Información sobre técnicas de pintura
    - Consejos y mejores prácticas
    - Guías paso a paso
    - Preguntas sobre cómo hacer algo
    
    Productos incluye:
    - Preguntas sobre productos específicos
    - Información sobre precios
    - Características de productos
    - Comparaciones entre productos
    - Disponibilidad de productos
    
    Mensaje: {last_message}
    
    Responde SOLO con 'guias' o 'productos'.
    """.format(
        last_message=last_message
    )

    result = state["llm"].predict(prompt).lower()
    logger.error(f"Select collection result: {result}")
    return {"next_step": "guias" if "guias" in result else "productos"}


def handle_small_talk(state: ChatState) -> ChatState:
    last_message = state["messages"][-1]["content"]
    prompt = """Responde al siguiente mensaje de manera amigable y profesional en español.
    Mantén un tono cordial y cercano, típico de una tienda de pinturas.
    
    Mensaje: {last_message}
    """.format(
        last_message=last_message
    )

    state["response"] = state["llm"].predict(prompt)
    state["next_step"] = "end"
    return state


def query_theory(state: ChatState) -> ChatState:
    embeddings = CohereEmbeddings(
        cohere_api_key=settings.COHERE_API_KEY, model="embed-multilingual-v3.0"
    )

    theory_store = Chroma(
        collection_name="guias",
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )

    # Create prompt template properly
    prompt_template = PromptTemplate(
        template="""Eres un experto en pinturas y técnicas de pintura.
        Usa la siguiente información para responder la pregunta del usuario en español.
        Si no encuentras la respuesta específica en el contexto, indica que no tienes esa información.
        
        Contexto: {context}
        
        Pregunta: {question}
        
        Respuesta:""",
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=state["llm"],
        chain_type="stuff",
        retriever=theory_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template},
    )

    state["response"] = chain.run(state["messages"][-1]["content"])
    state["next_step"] = "end"
    return state


def query_products(state: ChatState) -> ChatState:
    embeddings = CohereEmbeddings(
        cohere_api_key=settings.COHERE_API_KEY, model="embed-multilingual-v3.0"
    )

    products_store = Chroma(
        collection_name="productos",
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
    # Ver cantidad de documentos
    logger.info(f"Select collection result: {products_store._collection.count()}")
    logger.info(f"Colección: {products_store._collection.peek()}")

    # Create prompt template properly
    prompt_template = PromptTemplate(
        template="""Eres un experto en productos de pintura.
        Debes mostrar la inforación completa del producto que encuentres.
        Usa la siguiente información para responder la pregunta del usuario en español.
        Si no encuentras la respuesta específica en el contexto, indica que no tienes esa información.
        
        Contexto: {context}
        
        Pregunta: {question}
        
        Respuesta:""",
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=state["llm"],
        chain_type="stuff",
        retriever=products_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template},
    )

    state["response"] = chain.run(state["messages"][-1]["content"])
    state["next_step"] = "end"
    return state


def build_graph():
    workflow = StateGraph(ChatState)

    # Add nodes
    workflow.add_node("detect_intent", detect_intent)
    workflow.add_node("handle_small_talk", handle_small_talk)
    workflow.add_node("select_collection", select_collection)
    workflow.add_node("query_theory", query_theory)
    workflow.add_node("query_products", query_products)

    # Add edges with conditional logic
    workflow.set_entry_point("detect_intent")

    workflow.add_conditional_edges(
        "detect_intent",
        lambda x: x["next_step"],
        {
            "handle_small_talk": "handle_small_talk",
            "select_collection": "select_collection",
        },
    )

    workflow.add_conditional_edges(
        "select_collection",
        lambda x: x["next_step"],
        {"guias": "query_theory", "productos": "query_products"},
    )

    # Add end edges
    workflow.add_edge("handle_small_talk", END)
    workflow.add_edge("query_theory", END)
    workflow.add_edge("query_products", END)

    return workflow.compile()


class PaintChatbot:
    def __init__(self):
        self.graph = build_graph()
        self.sessions: Dict[str, List[Dict]] = {}
        self.llm = ChatCohere(
            cohere_api_key=settings.COHERE_API_KEY,
            model="command-r-plus-08-2024",
            seed=42,
            temperature=0.2,
            prompt_template="""Eres un asistente amable y profesional de una tienda de pinturas.
            Debes responder siempre en español, de manera clara y concisa.
            Tu objetivo es ayudar a los clientes con información sobre productos y guías de pintura.
            
            Mensaje del usuario: {message}
            """,
        )

    def process_message(self, user_id: str, message: str) -> str:
        try:
            if user_id not in self.sessions:
                self.sessions[user_id] = []

            self.sessions[user_id].append({"role": "user", "content": message})

            state = {
                "messages": self.sessions[user_id],
                "next_step": "start",
                "current_user": user_id,
                "response": "",
                "llm": self.llm,
            }

            end_state = self.graph.invoke(state)
            self.sessions[user_id].append(
                {"role": "assistant", "content": end_state["response"]}
            )

            return end_state["response"]

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise

    def get_chat_history(self, user_id: str) -> List[Dict]:
        return self.sessions.get(user_id, [])
