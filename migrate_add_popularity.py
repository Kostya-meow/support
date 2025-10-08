"""
Миграция: добавление поля popularity_score в knowledge_entries
"""
import sqlite3

def migrate():
    """Добавляет поле popularity_score в таблицу knowledge_entries"""
    conn = sqlite3.connect('knowledge.db')
    cursor = conn.cursor()
    
    try:
        # Проверяем, есть ли уже колонка
        cursor.execute("PRAGMA table_info(knowledge_entries)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'popularity_score' not in columns:
            cursor.execute("""
                ALTER TABLE knowledge_entries 
                ADD COLUMN popularity_score REAL DEFAULT 0.0
            """)
            conn.commit()
            print("✅ Поле popularity_score успешно добавлено")
        else:
            print("ℹ️  Поле popularity_score уже существует")
        
        # Создаем индекс для быстрой сортировки
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_popularity_score 
            ON knowledge_entries(popularity_score DESC)
        """)
        conn.commit()
        print("✅ Индекс для popularity_score создан")
        
    except Exception as e:
        print(f"❌ Ошибка миграции: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
