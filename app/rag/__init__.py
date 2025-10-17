"""
RAG (Retrieval-Augmented Generation) подсистема

Включает в себя:
- RAGService - основной сервис для работы с RAG
- KnowledgeBase - работа с базой знаний
- Retrieval функции
"""

from .service import (
    RAGService,
    RAGResult,
    ChatMessage,
    ToxicityClassifier,
    SpeechToTextService,
)
from .retrieval import KnowledgeBase

__all__ = [
    "RAGService",
    "RAGResult",
    "ChatMessage",
    "ToxicityClassifier",
    "SpeechToTextService",
    "KnowledgeBase",
]
