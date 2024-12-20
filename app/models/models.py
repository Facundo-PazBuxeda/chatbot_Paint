from enum import Enum
from typing import Optional, List, Dict
from pydantic import BaseModel

class IntentType(str, Enum):
    CHAT = "CHAT"
    RAG = "RAG"
    SQL = "SQL"
    UNKNOWN = "UNKNOWN"

class MessageIntent(BaseModel):
    type: IntentType
    confidence: float
    metadata: Optional[Dict] = {}

class Product(BaseModel):
    id: int
    marca: str
    nombre: str
    precio_regular: str
    precio_promo: str
    categoria_id: Optional[int]

class CustomException(Exception):
    def __init__(self, message: str, details: Optional[Dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class IntentClassificationError(CustomException):
    pass

class RAGProcessingError(CustomException):
    pass

class SQLQueryError(CustomException):
    pass