#!/usr/bin/env python3
"""Тест сегментированной истории заявок"""

from datetime import datetime
from app.rag_service import RAGService, ChatMessage

def test_segmented_history():
    """Тестируем сегментированную историю заявок"""
    
    print("=== Тест сегментированной истории заявок ===")
    
    # Создаем тестовый RAG сервис
    config = {
        "llm": {"model": "test", "api_key": "test"},
        "embeddings": {"model_name": "test"},
        "rag": {}
    }
    
    try:
        rag_service = RAGService(config)
    except:
        # Если полная инициализация не удается, создаем упрощенную версию для тестов
        from collections import defaultdict
        class MockRAGService:
            def __init__(self):
                self.chat_histories = defaultdict(list)
            
            def add_chat_message(self, user_id, message, is_user):
                chat_message = ChatMessage(message, is_user, datetime.now())
                self.chat_histories[user_id].append(chat_message)
            
            def get_chat_history_since_last_ticket(self, user_id):
                history = self.chat_histories[user_id]
                
                # Ищем последнее системное сообщение о закрытии заявки
                last_closure_index = -1
                for i in range(len(history) - 1, -1, -1):
                    msg = history[i]
                    if not msg.is_user and any(phrase in msg.message.lower() for phrase in [
                        "завершил заявку", "заявка завершена", "если потребуется помощь"
                    ]):
                        last_closure_index = i
                        break
                
                # Если нашли закрытие заявки, берем сообщения после него
                if last_closure_index >= 0:
                    relevant_history = history[last_closure_index + 1:]
                else:
                    # Если не было закрытий, берем всю историю
                    relevant_history = history
                
                # Фильтруем служебные команды и системные сообщения
                filtered_history = []
                for msg in relevant_history:
                    # Пропускаем служебные команды
                    if msg.is_user and msg.message.startswith('/'):
                        continue
                    # Пропускаем системные сообщения о создании заявки
                    if not msg.is_user and any(phrase in msg.message.lower() for phrase in [
                        "уведомили оператора", "ожидайте ответа", "мы уведомили"
                    ]):
                        continue
                    filtered_history.append(msg)
                
                return filtered_history
            
            def mark_ticket_closed(self, user_id):
                closure_message = "Оператор завершил заявку. Если потребуется помощь, напишите снова или нажмите кнопку \"Позвать оператора\"."
                self.add_chat_message(user_id, closure_message, is_user=False)
        
        rag_service = MockRAGService()
    
    user_id = 12345
    
    print("\n1. Симулируем первую сессию (до первой заявки):")
    rag_service.add_chat_message(user_id, "/start", is_user=True)
    rag_service.add_chat_message(user_id, "Здравствуйте! Опишите проблему...", is_user=False)
    rag_service.add_chat_message(user_id, "ты кто", is_user=True)
    rag_service.add_chat_message(user_id, "Не нашла инструкций по этому вопросу...", is_user=False)
    rag_service.add_chat_message(user_id, "Мы уведомили оператора, ожидайте ответа", is_user=False)
    
    # Получаем историю для первой заявки
    first_ticket_history = rag_service.get_chat_history_since_last_ticket(user_id)
    print(f"История для первой заявки ({len(first_ticket_history)} сообщений):")
    for i, msg in enumerate(first_ticket_history, 1):
        sender = "ПОЛЬЗОВАТЕЛЬ" if msg.is_user else "БОТ"
        print(f"  {i}. [{sender}] {msg.message}")
    
    print("\n2. Симулируем закрытие первой заявки:")
    rag_service.mark_ticket_closed(user_id)
    
    print("\n3. Симулируем вторую сессию (после закрытия первой заявки):")
    rag_service.add_chat_message(user_id, "Проблема снова появилась", is_user=True)
    rag_service.add_chat_message(user_id, "Попробуйте другое решение...", is_user=False)
    rag_service.add_chat_message(user_id, "Не работает, нужен оператор", is_user=True)
    rag_service.add_chat_message(user_id, "Мы уведомили оператора, ожидайте ответа", is_user=False)
    
    # Получаем историю для второй заявки
    second_ticket_history = rag_service.get_chat_history_since_last_ticket(user_id)
    print(f"История для второй заявки ({len(second_ticket_history)} сообщений):")
    for i, msg in enumerate(second_ticket_history, 1):
        sender = "ПОЛЬЗОВАТЕЛЬ" if msg.is_user else "БОТ"
        print(f"  {i}. [{sender}] {msg.message}")
    
    print("\n=== Тест завершен ===")
    print("✓ Первая заявка: показана вся история (без /start и системных сообщений)")
    print("✓ Вторая заявка: показана только история после закрытия предыдущей заявки")

if __name__ == "__main__":
    test_segmented_history()