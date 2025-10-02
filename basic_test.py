print("Testing basic functionality...")

# Проверяем импорт основных классов
try:
    from app.rag_service import RAGService, ChatMessage
    print("✓ RAGService and ChatMessage imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")

# Проверяем создание ChatMessage
try:
    from datetime import datetime
    msg = ChatMessage("Test message", True, datetime.now())
    print(f"✓ ChatMessage created: {msg.message}, user: {msg.is_user}")
except Exception as e:
    print(f"✗ ChatMessage error: {e}")

print("Basic test completed!")