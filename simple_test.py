import sys
print("Python version:", sys.version)
print("Test successful!")

# Простой тест ChatMessage
from datetime import datetime

class ChatMessage:
    def __init__(self, message: str, is_user: bool, timestamp: datetime):
        self.message = message
        self.is_user = is_user
        self.timestamp = timestamp

msg = ChatMessage("Тест", True, datetime.now())
print(f"Message: {msg.message}, User: {msg.is_user}, Time: {msg.timestamp}")
print("ChatMessage class works correctly!")