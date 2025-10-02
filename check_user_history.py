#!/usr/bin/env python3
"""Проверка истории конкретного пользователя"""

import asyncio
from app.rag_service import RAGService
from app.config import settings

async def check_user_history():
    """Проверяем историю для конкретного пользователя"""
    try:
        # Инициализируем RAG сервис
        rag_service = RAGService(settings.model_dump())
        await rag_service.prepare()
        
        # ID пользователя из Telegram (константин кожин)
        user_id = 473050506  # Это его ID из логов
        
        print(f"=== Проверка истории для пользователя {user_id} ===")
        
        # Проверяем полную историю
        full_history = rag_service.get_chat_history(user_id)
        print(f"\nПолная история ({len(full_history)} сообщений):")
        for i, msg in enumerate(full_history, 1):
            sender = "ПОЛЬЗОВАТЕЛЬ" if msg.is_user else "БОТ"
            timestamp = msg.timestamp.strftime("%H:%M:%S")
            print(f"  {i}. [{timestamp}] {sender}: {msg.message}")
        
        # Проверяем сегментированную историю
        segmented_history = rag_service.get_chat_history_since_last_ticket(user_id)
        print(f"\nСегментированная история ({len(segmented_history)} сообщений):")
        for i, msg in enumerate(segmented_history, 1):
            sender = "ПОЛЬЗОВАТЕЛЬ" if msg.is_user else "БОТ"
            timestamp = msg.timestamp.strftime("%H:%M:%S")
            print(f"  {i}. [{timestamp}] {sender}: {msg.message}")
            
        if len(segmented_history) == 0:
            print("⚠️  Сегментированная история пустая!")
            print("Возможные причины:")
            print("1. Сообщения не сохраняются в RAG сервисе")
            print("2. Логика фильтрации исключает все сообщения")
            print("3. Неправильный user_id")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_user_history())