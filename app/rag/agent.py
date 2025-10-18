"""
RAG Agent - интеллектуальный помощник с инструментами
"""
import logging
import os
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class RAGAgent:
    """Простой RAG агент с инструментами"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.categories = config.get("agent", {}).get("categories", [])
        
        # Инициализируем только если agno доступно
        self.agent = None
        try:
            self._init_agent()
        except ImportError:
            logger.warning("agno not available, using fallback mode")
    
    def _init_agent(self):
        """Инициализация agno агента"""
        from agno.agent import Agent
        from agno.models.llama_cpp import LlamaCpp
        from app.rag.agent_tools import (
            search_knowledge_base, 
            classify_request, 
            call_operator,
            get_system_status
        )
        
        # Настройка модели
        llm = LlamaCpp(
            id="gemma-3-27b-it",
            base_url="https://demo.ai.sfu-kras.ru/v1",
        )
        
        # Получаем системные инструкции из конфига
        system_instructions = self.config.get("agent", {}).get("system_instructions", 
            "Ты - помощник технической поддержки. Отвечай полезно и дружелюбно.")

        self.agent = Agent(
            name="RAG Support Agent",
            description="Агент технической поддержки с доступом к базе знаний",
            model=llm,
            tools=[search_knowledge_base, classify_request, call_operator, get_system_status],
            instructions=system_instructions,
            debug_mode=False,
            store_history_messages=False,
            store_tool_messages=False,
            store_media=False,
        )
    
    async def process_query(self, query: str) -> str:
        """Обработка запроса через LLM агента"""
        print(f"\n[AGENT START] Обрабатываю запрос: '{query}'")
        
        if not self.agent:
            print("⚠️ [AGENT] Агент недоступен, использую fallback")
            return self._fallback_process(query)
        
        try:
            print("[AGENT] Отправляю запрос в LLM...")
            
            # Отправляем запрос в LLM через агента (async для поддержки async инструментов)
            result = await self.agent.arun(query)
            
            # Извлекаем текст ответа
            if hasattr(result, 'content'):
                response = result.content.strip()
            else:
                response = str(result).strip()
            
            print(f"✅ [AGENT COMPLETE] Ответ получен, длина: {len(response)} символов")
            return response
            
        except Exception as e:
            print(f"💥 [AGENT ERROR] Ошибка агента: {e}")
            logger.error(f"Agent error: {e}")
            return self._fallback_process(query)
    
    def _fallback_process(self, query: str) -> str:
        """Простая обработка без агента - возвращаем стандартный ответ"""
        print(f"🔄 [FALLBACK] Обрабатываю запрос без агента: '{query}'")
        logger.info(f"Fallback processing query: {query}")
        return "Сейчас сервис недоступен. Попробуйте позже или обратитесь к оператору."