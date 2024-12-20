# services/memory_service.py
from typing import Dict, List, Optional
import time
from dataclasses import dataclass, field
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class Conversation:
    messages: List[str] = field(default_factory=list)
    last_activity: float = field(default_factory=time.time)

class MemoryService:
    def __init__(self, message_limit: int = 5, cleanup_interval: int = 3600):
        self.conversations: Dict[str, Conversation] = {}
        self.message_limit = message_limit
        self.cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """Inicia el servicio de memoria y el loop de limpieza"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Memory service cleanup task started")

    async def stop(self):
        """Detiene el servicio de memoria"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Memory service cleanup task stopped")

    def add_message(self, user_id: str, message: str):
        """Añade un mensaje a la conversación"""
        if user_id not in self.conversations:
            self.conversations[user_id] = Conversation()
        
        conversation = self.conversations[user_id]
        conversation.messages.append(message)
        conversation.last_activity = time.time()
        
        # Mantener solo los mensajes más recientes
        if len(conversation.messages) > self.message_limit:
            conversation.messages = conversation.messages[-self.message_limit:]

    def get_messages(self, user_id: str) -> List[str]:
        """Obtiene los mensajes de una conversación"""
        if user_id not in self.conversations:
            return []
        return self.conversations[user_id].messages.copy()

    def clear_messages(self, user_id: str):
        """Limpia los mensajes de una conversación"""
        if user_id in self.conversations:
            self.conversations[user_id] = Conversation()

    async def _cleanup_loop(self):
        """Loop de limpieza de conversaciones inactivas"""
        while True:
            try:
                current_time = time.time()
                to_remove = []
                
                for user_id, conversation in self.conversations.items():
                    if current_time - conversation.last_activity > self.cleanup_interval:
                        to_remove.append(user_id)
                
                for user_id in to_remove:
                    del self.conversations[user_id]
                    logger.info(f"Cleaned up conversation for user {user_id}")
                
                await asyncio.sleep(self.cleanup_interval)
                
            except asyncio.CancelledError:
                logger.info("Cleanup loop cancelled")
                raise
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(60)  # Esperar un minuto antes de reintentar