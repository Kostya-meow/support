#!/usr/bin/env python3
"""Тест функциональности сохранения истории чата"""

import asyncio
from app.rag_service import RAGService
from app.config import settings

async def test_chat_history():
    """Тестируем сохранение истории чата"""
    # Инициализируем RAG сервис
    rag_service = RAGService(settings.model_dump())
    
    # Симулируем пользователя с ID 12345
    user_id = 12345
    
    print("=== Тестируем сохранение истории чата ===")
    
    # Добавляем несколько сообщений в чат
    rag_service.add_chat_message(user_id, "Привет! У меня проблема с интернетом", is_user=True)
    rag_service.add_chat_message(user_id, "Здравствуйте! Опишите проблему подробнее", is_user=False)
    rag_service.add_chat_message(user_id, "Интернет работает медленно", is_user=True)
    rag_service.add_chat_message(user_id, "Попробуйте перезагрузить роутер", is_user=False)
    rag_service.add_chat_message(user_id, "Не помогло. Хочу поговорить с оператором", is_user=True)
    
    # Получаем историю чата
    history = rag_service.get_chat_history(user_id)
    
    print(f"История чата для пользователя {user_id}:")
    for i, msg in enumerate(history, 1):
        sender = "Пользователь" if msg.is_user else "Бот"
        timestamp = msg.timestamp.strftime("%H:%M:%S")
        print(f"{i}. [{timestamp}] {sender}: {msg.message}")
    
    print(f"\nВсего сообщений в истории: {len(history)}")
    
    # Тестируем очистку истории
    rag_service.clear_chat_history(user_id)
    empty_history = rag_service.get_chat_history(user_id)
    print(f"После очистки сообщений: {len(empty_history)}")
    
    print("\n=== Тестируем автоматическое сохранение через generate_reply ===")
    
    # Тестируем автоматическое сохранение через generate_reply
    await rag_service.prepare()  # Подготавливаем RAG систему
    
    # Симулируем диалог
    result1 = rag_service.generate_reply(user_id, "Привет")
    print(f"Ответ бота: {result1.final_answer}")
    
    result2 = rag_service.generate_reply(user_id, "ты кто")
    print(f"Ответ бота: {result2.final_answer}")
    
    # Проверяем историю
    auto_history = rag_service.get_chat_history(user_id)
    print(f"\nАвтоматически сохраненная история ({len(auto_history)} сообщений):")
    for i, msg in enumerate(auto_history, 1):
        sender = "Пользователь" if msg.is_user else "Бот"
        timestamp = msg.timestamp.strftime("%H:%M:%S")
        print(f"{i}. [{timestamp}] {sender}: {msg.message}")
    
    print("\n=== Тест завершен ===")

if __name__ == "__main__":
    asyncio.run(test_chat_history())