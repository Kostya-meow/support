#!/usr/bin/env python3
"""
Миграция базы данных для добавления поля is_archived в таблицу conversations
"""

import asyncio
import sqlite3
from pathlib import Path

async def migrate_database():
    """Добавляет поле is_archived в таблицу conversations если его нет"""
    db_path = Path("support.db")
    
    if not db_path.exists():
        print("База данных не найдена. Создайте её сначала запустив приложение.")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Проверяем существует ли поле is_archived
        cursor.execute("PRAGMA table_info(conversations)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'is_archived' not in columns:
            print("Добавляю поле is_archived в таблицу conversations...")
            cursor.execute("ALTER TABLE conversations ADD COLUMN is_archived BOOLEAN DEFAULT 0 NOT NULL")
            conn.commit()
            print("✅ Поле is_archived успешно добавлено!")
        else:
            print("✅ Поле is_archived уже существует в базе данных")
            
    except Exception as e:
        print(f"❌ Ошибка при миграции: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    asyncio.run(migrate_database())