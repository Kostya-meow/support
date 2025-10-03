"""
Тестовый скрипт для проверки изменений
"""

import requests

BASE_URL = "http://127.0.0.1:8000"

def test_faq_page():
    """Проверка FAQ страницы"""
    try:
        response = requests.get(f"{BASE_URL}/faq")
        if response.status_code == 200:
            print("✅ FAQ страница доступна")
            print(f"   Размер HTML: {len(response.text)} bytes")
        else:
            print(f"❌ FAQ страница: статус {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка подключения к FAQ: {e}")

def test_faq_api():
    """Проверка API FAQ"""
    try:
        response = requests.get(f"{BASE_URL}/api/faq")
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])
            print(f"✅ API FAQ работает")
            print(f"   Количество вопросов: {len(items)}")
            if items:
                print(f"   Топ-1: {items[0]['question'][:50]}... (популярность: {items[0]['popularity_score']})")
        else:
            print(f"❌ API FAQ: статус {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка подключения к API: {e}")

if __name__ == "__main__":
    print("\n🧪 Тестирование FAQ функционала\n")
    print("⚠️  Запустите сервер: .venv\\Scripts\\python.exe -m uvicorn app.main:app --reload\n")
    
    test_faq_page()
    test_faq_api()
    
    print("\n✨ Тестирование завершено!")
