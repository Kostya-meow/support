"""
Подсистема ботов для различных платформ

Включает в себя:
- telegram_bot - Telegram бот
- vk_bot - VK бот
- Общие функции и утилиты для ботов
"""

from .telegram_bot import create_dispatcher, start_bot
from .vk_bot import create_vk_bot, start_vk_bot

__all__ = ["create_dispatcher", "start_bot", "create_vk_bot", "start_vk_bot"]
