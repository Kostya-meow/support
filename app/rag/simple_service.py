"""
Простой RAG Service с интеграцией агента
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

from app.db import get_knowledge_session
from app.db import tickets_crud as knowledge_crud
from app.config import load_config
from app.rag.agent import RAGAgent

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.model = None
        self.config = load_config()
        self.agent = None
        self._initialize_model()
        self._initialize_agent()

    def _initialize_model(self):
        """Инициализация модели для создания embeddings"""
        try:
            model_name = self.config.get("rag", {}).get(
                "model", "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.model = SentenceTransformer(model_name)
            logger.info(f"RAG model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load RAG model: {e}")
            self.model = None

    def _initialize_agent(self):
        """Инициализация RAG агента"""
        try:
            self.agent = RAGAgent(self.config)
            logger.info("RAG Agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Agent: {e}")
            self.agent = None

    def create_embedding(self, text: str) -> Optional[List[float]]:
        """Создание эмбеддинга для текста"""
        if not self.model:
            return None

        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            return None

    async def process_query(self, query: str) -> str:
        """Основной метод обработки запроса через агента"""
        if self.agent:
            # Используем агента для обработки
            try:
                return self.agent.process_query(query)
            except Exception as e:
                logger.error(f"Agent processing failed: {e}")
                return await self._fallback_search(query)
        else:
            # Простой поиск без агента
            return await self._fallback_search(query)

    async def _fallback_search(self, query: str) -> str:
        """Простой поиск по базе знаний без агента"""
        try:
            async with get_knowledge_session()() as session:
                chunks = await knowledge_crud.load_all_chunks(session)

                if not chunks:
                    return "База знаний пуста. Обратитесь к оператору."

                # Простой поиск по ключевым словам
                query_lower = query.lower()
                best_match = None

                for chunk in chunks:
                    if query_lower in chunk.content.lower():
                        best_match = chunk
                        break

                if best_match:
                    content = (
                        best_match.content[:300] + "..."
                        if len(best_match.content) > 300
                        else best_match.content
                    )
                    return f"Найдено в базе знаний:\n\nИсточник: {best_match.source_file}\n\n{content}\n\nЕсли нужна дополнительная помощь, обратитесь к оператору."
                else:
                    return "Информация не найдена в базе знаний. Обратитесь к оператору для получения помощи."

        except Exception as e:
            logger.error(f"Fallback search error: {e}")
            return "Произошла ошибка при поиске. Обратитесь к оператору."


# Создаем глобальный экземпляр сервиса
rag_service = RAGService()


def get_rag_service() -> RAGService:
    """Получение экземпляра RAG сервиса"""
    return rag_service
