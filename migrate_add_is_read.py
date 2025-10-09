"""
Миграция: Добавление поля is_read в таблицу messages
"""
import sqlite3

def migrate():
    # Подключаемся к базе tickets
    conn = sqlite3.connect('tickets.db')
    cursor = conn.cursor()
    
    try:
        # Проверяем, существует ли уже колонка
        cursor.execute("PRAGMA table_info(messages)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'is_read' not in columns:
            print("Добавляем колонку is_read...")
            cursor.execute("ALTER TABLE messages ADD COLUMN is_read BOOLEAN DEFAULT 0 NOT NULL")
            
            # Отмечаем все сообщения от операторов как прочитанные
            cursor.execute("UPDATE messages SET is_read = 1 WHERE sender = 'operator'")
            
            conn.commit()
            print("✅ Миграция успешно выполнена!")
            print(f"   - Добавлена колонка is_read")
            print(f"   - Сообщения от операторов отмечены как прочитанные")
        else:
            print("⚠️  Колонка is_read уже существует, пропускаем миграцию")
            
    except Exception as e:
        print(f"❌ Ошибка миграции: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
