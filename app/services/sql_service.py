from langchain.agents.agent_types import AgentType
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_cohere import ChatCohere, create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from core.config import settings
from typing import List, Optional, Dict
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from models.models import SQLQueryError, Product
import logging
import json

logger = logging.getLogger(__name__)

class SQLService:
    def __init__(self, db_url: str, cohere_api_key: str):
        self.db = SQLDatabase.from_uri(db_url)
        self.llm = ChatCohere(
            cohere_api_key=cohere_api_key,
            model="command-r-plus-08-2024",
            temperature=0
        )

    async def get_response(self, query: str) -> str:
        try:
            if "precio" in query.lower() or "cuesta" in query.lower():
                sql = """
                SELECT marca, nombre, precio_regular, precio_promo 
                FROM productos 
                WHERE marca LIKE '%Alba%' OR marca LIKE '%Sherwin%'
                LIMIT 5
                """
            else:
                sql = """
                SELECT DISTINCT nombre,marca 
                FROM productos 
                LIMIT 5
                """
            
            logger.info(f"Executing SQL: {sql}")
            result = self.db.run(sql)
            
            if not result:
                return "No encontré productos que coincidan con tu búsqueda."
            
            response = "Productos encontrados:\n"
            for row in result:
                if len(row) > 2:
                    response += f"- {row[0]} {row[1]}: ${row[2]}"
                    if row[3]: 
                        response += f" (Promo: ${row[3]})"
                    response += "\n"
                else:
                    response += f"- {row[0]}\n"
                    
            return response
            
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return f"Error consultando la base de datos: {str(e)}"