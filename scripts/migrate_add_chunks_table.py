"""
Миграция: Создание таблицы document_chunks для новой системы хранения знаний
"""

import asyncio
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from app.db.database import KnowledgeSessionLocal
from app.db.models import KnowledgeBase


async def migrate():
    """Создание таблицы document_chunks"""
    
    async with KnowledgeSessionLocal() as session:
        # Создаём таблицу через SQLAlchemy
        async with session.begin():
            # Получаем engine из session
            from app.db.database import knowledge_engine
            
            # Создаём все таблицы, определённые в KnowledgeBase
            async with knowledge_engine.begin() as conn:
                await conn.run_sync(KnowledgeBase.metadata.create_all)
            
        print("✅ Таблица document_chunks успешно создана!")
        
        # Проверяем что таблица создалась
        result = await session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='document_chunks'")
        )
        table = result.scalar_one_or_none()
        
        if table:
            print(f"✅ Таблица '{table}' существует в базе данных")
        else:
            print("⚠️  Таблица не найдена, возможно произошла ошибка")


if __name__ == "__main__":
    print("Запуск миграции: создание таблицы document_chunks...")
    asyncio.run(migrate())
    print("Миграция завершена!")
