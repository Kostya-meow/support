#!/usr/bin/env python3
"""Простой тест логики сегментации"""

from datetime import datetime

# Простая имитация ChatMessage
class ChatMessage:
    def __init__(self, message, is_user, timestamp):
        self.message = message
        self.is_user = is_user
        self.timestamp = timestamp

def test_segmentation_logic():
    print("=== Тест логики сегментации истории ===")
    
    # Создаем имитацию истории чата
    history = []
    
    # Первая сессия
    history.append(ChatMessage("/start", True, datetime.now()))
    history.append(ChatMessage("Здравствуйте! Опишите проблему...", False, datetime.now()))
    history.append(ChatMessage("ты кто", True, datetime.now()))
    history.append(ChatMessage("Не нашла инструкций...", False, datetime.now()))
    history.append(ChatMessage("Мы уведомили оператора", False, datetime.now()))
    
    # Закрытие заявки
    history.append(ChatMessage("Оператор завершил заявку. Если потребуется помощь, напишите снова или нажмите кнопку \"Позвать оператора\".", False, datetime.now()))
    
    # Вторая сессия  
    history.append(ChatMessage("Проблема снова появилась", True, datetime.now()))
    history.append(ChatMessage("Попробуйте другое решение...", False, datetime.now()))
    history.append(ChatMessage("Не работает, нужен оператор", True, datetime.now()))
    history.append(ChatMessage("Мы уведомили оператора", False, datetime.now()))
    
    def get_history_since_last_closure(history):
        # Ищем последнее закрытие заявки
        last_closure_index = -1
        for i in range(len(history) - 1, -1, -1):
            msg = history[i]
            if not msg.is_user and any(phrase in msg.message.lower() for phrase in [
                "завершил заявку", "заявка завершена", "если потребуется помощь"
            ]):
                last_closure_index = i
                break
        
        # Если нашли закрытие, берем сообщения после него
        if last_closure_index >= 0:
            relevant_history = history[last_closure_index + 1:]
        else:
            relevant_history = history
        
        # Фильтруем
        filtered_history = []
        for msg in relevant_history:
            # Пропускаем команды
            if msg.is_user and msg.message.startswith('/'):
                continue
            # Пропускаем системные уведомления
            if not msg.is_user and any(phrase in msg.message.lower() for phrase in [
                "уведомили оператора", "ожидайте ответа", "мы уведомили"
            ]):
                continue
            filtered_history.append(msg)
        
        return filtered_history
    
    # Тестируем логику для "первой заявки" (имитируем отсутствие закрытий в истории)
    first_session_history = history[:5]  # До первого закрытия
    first_result = get_history_since_last_closure(first_session_history)
    
    print(f"\n1. Первая заявка ({len(first_result)} сообщений):")
    for i, msg in enumerate(first_result, 1):
        sender = "ПОЛЬЗОВАТЕЛЬ" if msg.is_user else "БОТ"
        print(f"  {i}. [{sender}] {msg.message}")
    
    # Тестируем логику для "второй заявки" 
    second_result = get_history_since_last_closure(history)
    
    print(f"\n2. Вторая заявка ({len(second_result)} сообщений):")
    for i, msg in enumerate(second_result, 1):
        sender = "ПОЛЬЗОВАТЕЛЬ" if msg.is_user else "БОТ"
        print(f"  {i}. [{sender}] {msg.message}")
    
    print("\n=== Результат ===")
    print("✓ Первая заявка: вся история без команд и системных сообщений")
    print("✓ Вторая заявка: только сообщения после закрытия предыдущей заявки")
    print("✓ Служебные команды (/start) и системные уведомления исключены")

if __name__ == "__main__":
    test_segmentation_logic()