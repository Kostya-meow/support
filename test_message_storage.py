#!/usr/bin/env python3
"""Тест сохранения сообщений в RAG сервисе"""

import asyncio
from app.rag_service import RAGService
from app.config import settings

async def test_message_storage():
    """Тестируем сохранение сообщений"""
    print("=== Тест сохранения сообщений в RAG ===")
    
    try:
        # Инициализируем RAG сервис
        rag_service = RAGService(settings.model_dump())
        await rag_service.prepare()
        
        user_id = 12345
        
        print(f"1. Тестируем generate_reply с пользователем {user_id}")
        
        # Симулируем диалог через generate_reply
        result1 = rag_service.generate_reply(user_id, "/start")
        print(f"Ответ на /start: {result1.final_answer[:50]}...")
        
        result2 = rag_service.generate_reply(user_id, "ты кто")
        print(f"Ответ на 'ты кто': {result2.final_answer[:50]}...")
        
        result3 = rag_service.generate_reply(user_id, "Как тебя зовут Костя")
        print(f"Ответ на 'Как тебя зовут Костя': {result3.final_answer[:50]}...")
        
        # Проверяем полную историю
        full_history = rag_service.get_chat_history(user_id)
        print(f"\n2. Полная история ({len(full_history)} сообщений):")
        for i, msg in enumerate(full_history, 1):
            sender = "ПОЛЬЗОВАТЕЛЬ" if msg.is_user else "БОТ"
            print(f"  {i}. [{sender}] {msg.message[:80]}...")
        
        # Проверяем сегментированную историю
        segmented_history = rag_service.get_chat_history_since_last_ticket(user_id)
        print(f"\n3. Сегментированная история ({len(segmented_history)} сообщений):")
        for i, msg in enumerate(segmented_history, 1):
            sender = "ПОЛЬЗОВАТЕЛЬ" if msg.is_user else "БОТ"
            print(f"  {i}. [{sender}] {msg.message[:80]}...")
        
        print("\n4. Симулируем закрытие заявки")
        rag_service.mark_ticket_closed(user_id)
        
        print("5. Добавляем новые сообщения после закрытия")
        result4 = rag_service.generate_reply(user_id, "Проблема появилась снова")
        print(f"Ответ на новую проблему: {result4.final_answer[:50]}...")
        
        # Проверяем сегментированную историю после закрытия
        new_segmented_history = rag_service.get_chat_history_since_last_ticket(user_id)
        print(f"\n6. Новая сегментированная история ({len(new_segmented_history)} сообщений):")
        for i, msg in enumerate(new_segmented_history, 1):
            sender = "ПОЛЬЗОВАТЕЛЬ" if msg.is_user else "БОТ"
            print(f"  {i}. [{sender}] {msg.message[:80]}...")
        
        print("\n=== Тест завершен ===")
        
    except Exception as e:
        print(f"Ошибка в тесте: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_message_storage())